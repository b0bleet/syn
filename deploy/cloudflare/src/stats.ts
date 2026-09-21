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
    return { body: render(Object.fromEntries(names.map(([name], i) => [name, results[i]]))), ok: true };
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

function render(data: Record<string, Row[]>): string {
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
<script>
// Swap in fresh numbers every 30 s while the page is in view, for half an hour.
let left = 60;
const timer = setInterval(async () => {
  if (document.hidden) return;
  if (--left === 0) clearInterval(timer);
  const response = await fetch(location.href, { cache: "no-cache" }).catch(() => null);
  if (!response?.ok) return;
  const next = new DOMParser().parseFromString(await response.text(), "text/html").querySelector("main");
  if (next) document.querySelector("main").replaceWith(next);
}, 30000);
</script>
</body></html>`;
}
