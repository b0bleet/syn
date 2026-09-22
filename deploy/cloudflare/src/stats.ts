/**
 * The public statistics page at /stats: today's counts, live from the LiveStats Durable Object,
 * then tables for a range from the `syn_api_calls` Analytics Engine dataset, read through
 * Cloudflare's SQL API. The Worker needs STATS_ACCOUNT_ID and STATS_API_TOKEN (a token with only
 * "Account Analytics: Read") for the tables; the token never leaves the Worker.
 *
 * Tables run one query per section and are refreshed at most every ten minutes, in LiveStats so
 * every data center shares them. The page itself is cached for 30 seconds per data center, and
 * an open page refreshes itself every 30 seconds for half an hour: each refresh is a Worker
 * request, and the free plan's 100,000 a day are shared with the API. Counts are weighted by
 * _sample_interval, so they stay right if Cloudflare samples at high volume. Client IDs rotate
 * daily, so distinct users are per day only. Calling websites are recorded but not shown: any
 * script can claim any Origin, so a public list would print whatever a spammer sends.
 */

import type { LiveStats, Today } from "./live";

export interface StatsEnv {
  STATS_ACCOUNT_ID?: string;
  STATS_API_TOKEN?: string;
}

/** A range's tables, rendered, as of `at` (epoch ms). */
export interface Tables {
  at: number;
  ok: boolean;
  body: string;
}

type Row = Record<string, string | number>;

const DATASET = "syn_api_calls";
export const RANGES = [1, 7, 30, 90];
// Seconds a data center reuses a render of the page.
const EDGE_S = 30;
const UNAVAILABLE = "<p>Statistics are unavailable right now. Try again in a minute.</p>";
const CALLS = "sum(_sample_interval) AS calls";
const TEXTS = "sum(_sample_interval * double1) AS texts";
const P50 = "quantileExactWeighted(0.5)(double2, _sample_interval) AS p50_ms";
const P95 = "quantileExactWeighted(0.95)(double2, _sample_interval) AS p95_ms";
const KEYED = "sumIf(_sample_interval, blob3 = 'key') AS keyed";
const LIMITED = "sumIf(_sample_interval, blob2 = '429') AS limited";
const FAILED = "sumIf(_sample_interval, blob2 >= '500') AS failed";

/** One query per section, over the last `days` days. */
export function queries(days: number): Record<string, string> {
  const from = `FROM ${DATASET} WHERE timestamp > now() - INTERVAL '${days}' DAY`;
  return {
    totals: `SELECT ${CALLS}, ${TEXTS}, ${KEYED}, ${LIMITED}, ${FAILED}, ${P50}, ${P95} ${from}`,
    days: `SELECT toStartOfDay(timestamp) AS day, ${CALLS}, ${TEXTS},
             count(DISTINCT index1) AS users, ${KEYED}, ${LIMITED}, ${FAILED}, ${P50}, ${P95}
           ${from} GROUP BY day ORDER BY day DESC`,
    // Same window as the other sections. The chart groups bars when a day range is wider than
    // the drawing can show one bar per hour.
    hours: `SELECT toStartOfInterval(timestamp, INTERVAL '1' HOUR) AS hour, ${CALLS}
            ${from} GROUP BY hour ORDER BY hour`,
    endpoints: `SELECT blob1 AS endpoint, ${CALLS}, ${TEXTS},
                  sumIf(_sample_interval, blob2 >= '400') AS errors, ${P50}
                ${from} GROUP BY endpoint ORDER BY calls DESC`,
    statuses: `SELECT blob2 AS status, ${CALLS} ${from} GROUP BY status ORDER BY calls DESC`,
    sources: `SELECT blob4 AS source, blob5 AS client, ${CALLS} ${from}
              GROUP BY source, client ORDER BY calls DESC`,
    countries: `SELECT blob6 AS country, ${CALLS} ${from}
                GROUP BY country ORDER BY calls DESC LIMIT 20`,
    gpu: `SELECT ${CALLS}, sum(_sample_interval * double4) / 1000 AS gpu_seconds,
            quantileExactWeighted(0.5)(double3, _sample_interval) AS wait_p50_ms,
            sumIf(_sample_interval, double3 > 10000) AS waited_over_10s
          ${from} AND double4 > 0`,
  };
}

async function query(env: StatsEnv, sql: string): Promise<Row[]> {
  const url = `https://api.cloudflare.com/client/v4/accounts/${env.STATS_ACCOUNT_ID}/analytics_engine/sql`;
  const response = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.STATS_API_TOKEN}` },
    body: `${sql} FORMAT JSON`,
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`Cloudflare answered ${response.status}: ${text.slice(0, 300)}`);
  return (JSON.parse(text) as { data: Row[] }).data;
}

export async function statsPage(url: URL, live: DurableObjectStub<LiveStats>): Promise<Response> {
  const asked = Number(url.searchParams.get("days"));
  const days = RANGES.includes(asked) ? asked : 7;
  // One cache entry per range, whatever else the query string holds.
  const key = new Request(`${url.origin}/stats?days=${days}`);
  const cache = typeof caches === "undefined" ? undefined : caches.default;
  const cached = await cache?.match(key);
  if (cached) return cached;
  let body: string;
  try {
    const { today, tables } = await live.page(days);
    body = main(days, today, tables);
  } catch (error) {
    console.error("Statistics unavailable", error);
    body = `<main>${UNAVAILABLE}</main>`;
  }
  const response = new Response(page(days, body), {
    headers: {
      "content-type": "text/html; charset=utf-8",
      "cache-control": `public, max-age=${EDGE_S}`,
    },
  });
  await cache?.put(key, response.clone());
  return response;
}

/** A range's tables, as LiveStats keeps them. Never throws: a failure is a message to show. */
export async function queryTables(days: number, env: StatsEnv): Promise<Tables> {
  return { at: Date.now(), ...(await content(days, env)) };
}

async function content(days: number, env: StatsEnv): Promise<{ body: string; ok: boolean }> {
  if (!env.STATS_ACCOUNT_ID || !env.STATS_API_TOKEN) {
    return { body: "<p>Statistics aren't connected yet.</p>", ok: false };
  }
  try {
    const names = Object.entries(queries(days));
    const results = await Promise.all(names.map(([, sql]) => query(env, sql)));
    return { body: render(Object.fromEntries(names.map(([name], i) => [name, results[i]])), days), ok: true };
  } catch (error) {
    // The details can name the account; they belong in the logs, not on a public page.
    console.error("Stats query failed", error);
    return { body: UNAVAILABLE, ok: false };
  }
}

function label(days: number): string {
  return days === 1 ? "24 hours" : `${days} days`;
}

/** The part an open page swaps in when it refreshes: today's counts, then the range's tables. */
function main(days: number, today: Today, tables: Tables): string {
  const updated = tables.ok ? `, updated ${new Date(tables.at).toISOString().slice(11, 16)} UTC` : "";
  return `<main><h2>Today (UTC), live</h2>${tiles([
    ["calls", today.calls],
    ["texts classified", today.texts],
    ["users", today.users],
    ["with an unlimited key", today.keyed],
    ["hit the daily limit", today.limited],
    ["failed", today.failed],
  ])}<h2>Last ${label(days)}${updated}</h2>${tables.body}</main>`;
}

function tiles(values: [string, string | number | undefined][]): string {
  return `<div class="tiles">${values.map(([name, value]) => `<div><b>${count(value)}</b>${name}</div>`).join("")}</div>`;
}

function render(data: Record<string, Row[]>, days: number): string {
  const total = data.totals[0] ?? {};
  if (!Number(total.calls)) return "<p>No API calls in this period.</p>";
  return [
    tiles([
      ["calls", total.calls],
      ["texts classified", total.texts],
      ["with an unlimited key", total.keyed],
      ["hit the daily limit", total.limited],
      ["failed", total.failed],
      ["ms median response", total.p50_ms],
      ["ms for the slowest 5%", total.p95_ms],
    ]),
    usageChart("Calls per day (UTC)", buckets(data.days ?? [], "day", 86_400_000, days)),
    usageChart("Calls per hour (UTC)", buckets(data.hours ?? [], "hour", 3_600_000, days)),
    section("Per day (UTC)", data.days, "calls", {
      day: "day",
      calls: "calls",
      texts: "texts",
      users: "users",
      keyed: "with key",
      limited: "limited",
      failed: "failed",
      p50_ms: "median ms",
      p95_ms: "p95 ms",
    }),
    section("Endpoints", data.endpoints, "calls", {
      endpoint: "endpoint",
      calls: "calls",
      texts: "texts",
      errors: "errors",
      p50_ms: "median ms",
    }),
    section("Where calls come from", data.sources, "calls", {
      source: "source",
      client: "client",
      calls: "calls",
    }),
    section("Countries", data.countries, "calls", { country: "country", calls: "calls" }),
    section("Status codes", data.statuses, "calls", { status: "status", calls: "calls" }),
    section("GPU", data.gpu, "", {
      calls: "jobs",
      gpu_seconds: "GPU seconds",
      wait_p50_ms: "median wait ms",
      waited_over_10s: "waited over 10 s (cold starts)",
    }),
  ].join("");
}

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;
// About one bar per four pixels on the 640-wide chart. Wider ranges sum neighboring hours.
const MAX_BARS = 192;

/** UTC buckets across the selected window, empty hours and days included, then summed to fit. */
function buckets(rows: Row[], key: string, step: number, days: number, now = Date.now()): { t: number; n: number; hours: number }[] {
  const start = Math.floor((now - days * DAY_MS) / step) * step;
  const end = Math.floor(now / step) * step;
  const values = new Map<number, number>();
  for (const row of rows) {
    const t = parseUtc(String(row[key] ?? ""));
    if (Number.isNaN(t)) continue;
    const bucket = Math.floor(t / step) * step;
    values.set(bucket, (values.get(bucket) ?? 0) + (Number(row.calls) || 0));
  }
  const points: { t: number; n: number; hours: number }[] = [];
  for (let t = start; t <= end; t += step) points.push({ t, n: values.get(t) ?? 0, hours: step / HOUR_MS });
  if (points.length <= MAX_BARS) return points;
  const size = Math.ceil(points.length / MAX_BARS);
  const grouped: { t: number; n: number; hours: number }[] = [];
  for (let i = 0; i < points.length; i += size) {
    const slice = points.slice(i, i + size);
    grouped.push({
      t: slice[0].t,
      n: slice.reduce((sum, point) => sum + point.n, 0),
      hours: slice.reduce((sum, point) => sum + point.hours, 0),
    });
  }
  return grouped;
}

function parseUtc(value: string): number {
  const iso = value.includes("T") ? value : value.replace(" ", "T");
  return Date.parse(/Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
}

/** Copper toward tide, one color per bar. Time is the only thing the color encodes. */
function barInk(index: number, total: number): string {
  const t = total <= 1 ? 0 : index / (total - 1);
  return `hsl(${(22 + t * 146).toFixed(0)} 54% 42%)`;
}

function tipText(point: { t: number; n: number; hours: number }): string {
  const date = new Date(point.t);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const day = `${date.getUTCDate()} ${months[date.getUTCMonth()]}`;
  const hour = String(date.getUTCHours()).padStart(2, "0");
  const calls = `${count(point.n)} ${point.n === 1 ? "call" : "calls"}`;
  if (point.hours === 24) return `${day} UTC · ${calls}`;
  if (point.hours === 1) return `${day} ${hour}:00 UTC · ${calls}`;
  return `${day} ${hour}:00 UTC, ${point.hours} hours · ${calls}`;
}

function stamp(t: number, withHour: boolean): string {
  const date = new Date(t);
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  if (!withHour) return `${month}-${day}`;
  return `${month}-${day} ${String(date.getUTCHours()).padStart(2, "0")}:00`;
}

/** A bar chart. `points` are already in time order. */
function usageChart(title: string, points: { t: number; n: number; hours: number }[]): string {
  if (!points.length) return "";
  const block = points[0].hours;
  const caption = block === 1 || block === 24 ? title : `${title}, ${block}-hour bars`;
  const max = Math.max(...points.map((point) => point.n), 1);
  const width = 640;
  const height = 156;
  const padL = 44;
  const padR = 4;
  const padT = 8;
  const padB = 22;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;
  const gap = points.length > 48 ? 0.4 : 1.5;
  const slot = innerW / points.length;
  const barW = Math.max(slot - gap, 0.3);
  const baseline = padT + innerH;
  const bars = points
    .map((point, i) => {
      const barH = (point.n / max) * innerH;
      const x = padL + i * slot + (slot - barW) / 2;
      const y = baseline - barH;
      const tip = escape(tipText(point));
      return `<g><rect class="mark" x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barW.toFixed(2)}" height="${barH.toFixed(2)}" fill="${barInk(i, points.length)}"/><rect class="hit" data-tip="${tip}" x="${(padL + i * slot).toFixed(2)}" y="${padT}" width="${slot.toFixed(2)}" height="${innerH}"/></g>`;
    })
    .join("");
  const ticks = [0, Math.floor((points.length - 1) / 2), points.length - 1];
  const labels = [...new Set(ticks)]
    .map((i) => {
      const x = padL + ((i + 0.5) * innerW) / points.length;
      const anchor = i === 0 ? "start" : i === points.length - 1 ? "end" : "middle";
      return `<text x="${x.toFixed(1)}" y="${height - 4}" text-anchor="${anchor}">${escape(stamp(points[i].t, block < 24))}</text>`;
    })
    .join("");
  const described = `${caption}. Highest bar ${count(max)} calls.`;
  return `<figure class="chart"><figcaption>${escape(caption)}</figcaption><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(described)}"><text x="0" y="${padT + 9}">${count(max)}</text><line x1="${padL}" y1="${baseline}" x2="${width - padR}" y2="${baseline}"/>${bars}${labels}</svg></figure>`;
}

/** A table, with a bar in the `bar` column scaled to its largest value. */
function section(title: string, rows: Row[], bar: string, columns: Record<string, string>): string {
  if (!rows.length) return `<h2>${title}</h2><p class="none">None in this period.</p>`;
  const max = Math.max(...rows.map((row) => Number(row[bar]) || 0), 1);
  const head = Object.values(columns).map((name) => `<th>${name}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = Object.keys(columns).map((key) => {
        const value = row[key];
        const shown = key === "day" ? escape(String(value).slice(0, 10)) : count(value);
        if (key !== bar) return `<td>${shown}</td>`;
        const width = ((Number(value) || 0) / max) * 100;
        return `<td class="bar"><span style="width:${width.toFixed(1)}%"></span>${shown}</td>`;
      });
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
  return `<h2>${title}</h2><table><tr>${head}</tr>${body}</table>`;
}

/** Whole numbers with separators; text is escaped, empty becomes a dash. */
function count(value: string | number | undefined): string {
  if (value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return escape(String(value));
  return Math.round(number).toLocaleString("en-US");
}

function escape(text: string): string {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function page(days: number, body: string): string {
  const links = RANGES.map((d) =>
    d === days ? `<b>${label(d)}</b>` : `<a href="?days=${d}">${label(d)}</a>`,
  ).join(" ");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sifty API statistics</title>
<meta name="description" content="Live usage of the sifty free text classification API: calls, texts classified, countries, and response times.">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<style>
:root { font: 13px/1.6 "Lucida Console", Monaco, monospace; color: #000; background: #fff; }
body { max-width: 1080px; margin: auto; padding: 32px 40px; }
header { display: flex; justify-content: space-between; align-items: baseline; gap: 20px; }
h1 { font-size: 20px; font-weight: normal; margin: 0; }
nav { display: flex; gap: 16px; font-size: 12px; }
a { color: inherit; text-underline-offset: 3px; }
.intro { font-size: 12px; margin: 12px 0 0; }
h2 { font-size: 13px; font-weight: bold; margin: 32px 0 8px; }
.chart { margin: 20px 0 0; }
.chart figcaption { font-size: 12px; margin: 0 0 6px; }
.chart svg { width: 100%; height: auto; display: block; }
.chart .hit { fill: transparent; cursor: crosshair; }
.chart g:hover .mark { fill: #000; }
.chart line { stroke: #000; }
.chart text { font: 10px "Lucida Console", Monaco, monospace; fill: #000; }
#chart-tip { position: fixed; z-index: 2; pointer-events: none; background: #fff; color: #000; border: 1px solid #000; padding: 4px 8px; font-size: 12px; white-space: nowrap; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; }
.tiles div { border: 1px solid #000; padding: 12px; font-size: 11px; }
.tiles b { display: block; font-size: 22px; font-weight: normal; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th { text-align: left; font-weight: normal; border-bottom: 1px solid #000; padding: 4px 8px 4px 0; }
td { padding: 4px 8px 4px 0; border-bottom: 1px solid #ddd; white-space: nowrap; }
td.bar { position: relative; width: 30%; }
td.bar span { position: absolute; left: 0; top: 5px; bottom: 5px; background: #000; opacity: .12; }
.none { font-size: 12px; }
@media (max-width: 700px) { body { padding: 20px; } header { flex-direction: column; } td, th { white-space: normal; } }
</style></head><body>
<header><h1><a href="/">sifty</a> API statistics</h1><nav>${links}</nav></header>
<p class="intro">Live usage of the free API: today's counts as they happen, tables every 10 minutes. Only counts are recorded: no text, labels, or IP addresses.</p>
${body}
<footer style="font-size:10px;margin-top:32px"><a href="/contact.html">Contact us</a> at <a href="mailto:emin@jolo.build">emin@jolo.build</a></footer>
<div id="chart-tip" hidden></div>
<script>
const tip = document.querySelector("#chart-tip");
function placeTip(event) {
  const hit = event.target.closest?.("[data-tip]");
  if (!hit) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.textContent = hit.getAttribute("data-tip");
  const box = tip.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
  const y = Math.max(8, event.clientY - box.height - 12);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
document.addEventListener("pointermove", placeTip);
document.addEventListener("pointerdown", placeTip);
// Swap in fresh numbers every 30 s while the page is in view, for half an hour.
let left = 60;
const timer = setInterval(async () => {
  if (document.hidden) return;
  if (--left === 0) clearInterval(timer);
  const response = await fetch(location.href, { cache: "no-cache" }).catch(() => null);
  if (!response?.ok) return;
  const next = new DOMParser().parseFromString(await response.text(), "text/html").querySelector("main");
  if (next) { tip.hidden = true; document.querySelector("main").replaceWith(next); }
}, 30000);
</script>
</body></html>`;
}
