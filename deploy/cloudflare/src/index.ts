/**
 * Public API and web page for syn on Cloudflare Workers.
 *
 * Holds no API logic. It meters the request, wraps it as the RunPod job input
 * {"http": {method, path, headers, body}}, waits for the GPU worker (polling through cold
 * starts), and returns the response the Python app produced there.
 *
 * Free for everyone: requests without a known key are counted per client IP per UTC day, and
 * against a global daily cap that bounds GPU spend. Keys listed in API_KEYS skip both. `/` is the
 * page in public/, and its files (icons, preview image, robots.txt) are served as they are;
 * neither is counted or reaches a GPU. /health reports RunPod worker counts without waking a GPU.
 * Every API call is recorded, without its content, in the STATS dataset and today's LIVE counts,
 * after its answer is sent; /stats shows both publicly.
 * REQUEST_LOGS additionally records request content in private Cloudflare Workers Logs.
 */

import type { LiveStats } from "./live";
import type { Quota } from "./quota";
import { logRequest } from "./request-log";
import { type StatsEnv, statsPage } from "./stats";

export { LiveStats } from "./live";
export { Quota } from "./quota";

export interface Env extends StatsEnv {
  /** Secret: the RunPod serverless endpoint running deploy/runpod/handler.py. */
  RUNPOD_ENDPOINT_ID: string;
  /** Secret: the RunPod API key. */
  RUNPOD_API_KEY: string;
  /** Secret: comma-separated keys that skip the free-tier limits. */
  API_KEYS?: string;
  /** Secret: sending-only Resend key used by the public contact form. */
  RESEND_API_KEY?: string;
  /** Private structured Workers Logs, with the Workers Free plan's three-day retention. */
  REQUEST_LOGS?: string;
  QUOTA: DurableObjectNamespace<Quota>;
  ASSETS: Fetcher;
  /** Free units per client IP per UTC day; one unit per request, or per text in a batch. */
  DAILY_LIMIT?: string;
  /** Free units per UTC day across all clients, so GPU spend has a ceiling. */
  GLOBAL_DAILY_LIMIT?: string;
  /** Whole-job deadline including a cold start. Free plan: keep polls under 50 subrequests. */
  JOB_TIMEOUT_SECONDS?: string;
  POLL_SECONDS?: string;
  /** Defaults to RunPod; override only to test against a local stand-in. */
  RUNPOD_API_BASE?: string;
  /** Analytics Engine dataset with one data point per API call, shown at /stats. */
  STATS?: AnalyticsEngineDataset;
  /** Today's counts, live, and the shared copy of the /stats tables. */
  LIVE: DurableObjectNamespace<LiveStats>;
}

interface Job {
  id?: string;
  status?: string;
  output?: unknown;
  error?: unknown;
  /** Milliseconds RunPod held the job before a GPU worker took it, cold start included. */
  delayTime?: number;
  /** Milliseconds the GPU worker spent on the job. */
  executionTime?: number;
}

interface ApiCall {
  response: Response;
  units: number;
  keyed: boolean;
  job?: Job;
}

interface HttpOutput {
  status: number;
  headers: Record<string, string>;
  body: string;
}

interface Charge {
  units: number;
  day: string;
  client: DurableObjectStub<Quota>;
  global: DurableObjectStub<Quota>;
  headers: Record<string, string>;
}

/**
 * Every file in public/ other than index.html; keep in step with that directory. Any other path
 * is an API call, so a file missing here would run a GPU job and cost the caller a unit.
 */
export const STATIC_FILES = new Set([
  "/robots.txt",
  "/sitemap.xml",
  "/llms.txt",
  "/site.webmanifest",
  "/og.png",
  "/favicon.ico",
  "/favicon.svg",
  "/apple-touch-icon.png",
  "/icon-192.png",
  "/icon-512.png",
]);
// Command-line clients get the app's plain-text usage at `/` unless they ask for HTML.
const COMMAND_LINE = /^(curl|wget|httpie|xh)\//i;
const PENDING = new Set(["IN_QUEUE", "IN_PROGRESS"]);
// What callers see when the GPU side fails; the specifics go to the Worker's logs.
const UNAVAILABLE = "The scoring service is temporarily unavailable. Please try again shortly.";
const CONTACT_TO = "emin@jolo.build";
const CONTACT_FROM = "sifty <contact@send.sifty.dev>";
const CONTACT_LIMIT = 5;
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Expose-Headers":
    "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, X-Syn-Selected, " +
    "X-Syn-Best, X-Syn-Confidence, X-Syn-Agreement, X-Syn-Abstain-Reasons, " +
    "x-typesafe-request-id",
};

class JobError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          ...CORS,
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Authorization, Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }
    if (url.pathname === "/health") return withCors(await health(env));
    if (url.pathname === "/contact" && request.method === "POST") {
      return contact(request, env, url);
    }
    const reading = request.method === "GET" || request.method === "HEAD";
    // Cloudflare Assets canonicalizes contact.html to /contact, so handle both names before the
    // catch-all API route rather than treating the clean URL as a classification request.
    if (reading && (url.pathname === "/contact" || url.pathname === "/contact.html")) {
      return env.ASSETS.fetch(request);
    }
    if (reading && url.pathname === "/stats") return statsPage(url, liveStats(env));
    if (reading && STATIC_FILES.has(url.pathname)) return env.ASSETS.fetch(request);
    if (reading && url.pathname === "/" && !url.search && wantsPage(request)) {
      return env.ASSETS.fetch(request);
    }
    const started = Date.now();
    const body = request.method === "GET" || request.method === "HEAD" ? null : await request.text();
    const logInfo = env.REQUEST_LOGS === "true" ? {
      request_id: crypto.randomUUID(),
      endpoint: routeOf(request.method, url.pathname, url.search),
      client: clientKind(request.headers.get("User-Agent") ?? ""),
    } : null;
    // Capture before any quota/GPU awaits: a cancelled invocation may never reach completion.
    if (logInfo) logRequest(request, url, body, started, { ...logInfo, phase: "received" });
    const call = await api(request, env, url, body);
    const ms = Date.now() - started;
    if (logInfo) {
      logRequest(request, url, body, started, {
        ...logInfo,
        phase: "completed",
        status: call.response.status,
        tier: call.keyed ? "key" : "free",
        units: call.units,
        duration_ms: ms,
        queue_ms: call.job?.delayTime,
        gpu_ms: call.job?.executionTime,
        response_body: isHttpOutput(call.job?.output) ? call.job.output.body : undefined,
        response_is_json: call.response.headers.get("Content-Type")?.includes("json") ?? false,
      });
    }
    // Recorded after the answer is sent, so it never slows or costs a caller their answer.
    ctx.waitUntil(
      record(request, url, call, ms, env).catch((error) =>
        console.error("API call not recorded", error),
      ),
    );
    return call.response;
  },
};

interface ContactMessage {
  name?: unknown;
  email?: unknown;
  message?: unknown;
  company?: unknown;
}

/** Validate a same-site contact submission and relay it through Resend without exposing the key. */
async function contact(request: Request, env: Env, url: URL): Promise<Response> {
  const origin = request.headers.get("Origin");
  if (origin && origin !== url.origin) return json({ detail: "This form must be sent from sifty." }, 403);
  if (!(request.headers.get("content-type") ?? "").toLowerCase().startsWith("application/json")) {
    return json({ detail: "Send the form as JSON." }, 415);
  }
  if (Number(request.headers.get("content-length") ?? 0) > 12_000) {
    return json({ detail: "The message is too long." }, 413);
  }

  let data: ContactMessage;
  try {
    const raw = await request.text();
    if (raw.length > 12_000) return json({ detail: "The message is too long." }, 413);
    data = JSON.parse(raw) as ContactMessage;
  } catch {
    return json({ detail: "The form could not be read." }, 400);
  }

  // A hidden field catches simple bots. Pretend it worked so they do not adapt and retry.
  if (typeof data.company === "string" && data.company.trim()) return json({ ok: true });
  const name = typeof data.name === "string" ? data.name.trim() : "";
  const email = typeof data.email === "string" ? data.email.trim() : "";
  const message = typeof data.message === "string" ? data.message.trim() : "";
  if (!name || name.length > 100) return json({ detail: "Enter your name (up to 100 characters)." }, 400);
  if (!validEmail(email)) return json({ detail: "Enter a valid email address." }, 400);
  if (!message || message.length > 5_000) {
    return json({ detail: "Enter a message (up to 5,000 characters)." }, 400);
  }
  if (!env.RESEND_API_KEY) {
    console.error("Contact form is missing RESEND_API_KEY");
    return json({ detail: "The contact form is unavailable right now. Please email us directly." }, 503);
  }

  const now = new Date();
  const day = now.toISOString().slice(0, 10);
  const resetAt = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
  const client = env.QUOTA.get(env.QUOTA.idFromName(`contact:${await clientId(request)}`));
  const allowance = await client.take(1, CONTACT_LIMIT, day, resetAt);
  if (!allowance.allowed) {
    const retryAfter = String(Math.ceil((resetAt - now.getTime()) / 1000));
    return json({ detail: "Too many messages today. Please email us directly." }, 429, {
      "Retry-After": retryAfter,
    });
  }

  const safeName = name.replace(/[\r\n]+/g, " ");
  const digest = await sha256(`${day}\n${await clientId(request)}\n${name}\n${email}\n${message}`);
  const idempotencyKey = `contact/${Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("")}`;
  let response: Response | null = null;
  try {
    response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.RESEND_API_KEY}`,
        "Content-Type": "application/json",
        "Idempotency-Key": idempotencyKey,
      },
      body: JSON.stringify({
        from: CONTACT_FROM,
        to: [CONTACT_TO],
        reply_to: email,
        subject: `sifty contact from ${safeName}`,
        text: `Name: ${name}\nEmail: ${email}\n\n${message}`,
      }),
    });
  } catch (error) {
    console.error("Resend unreachable", error);
  }
  if (!response?.ok) {
    console.error("Resend did not accept contact email", response?.status ?? "unreachable");
    return json({ detail: "The message could not be sent. Please email us directly." }, 502);
  }
  return json({ ok: true });
}

function validEmail(value: string): boolean {
  return value.length <= 254 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

/** The one LiveStats object, shared by every data center. */
function liveStats(env: Env): DurableObjectStub<LiveStats> {
  return env.LIVE.get(env.LIVE.idFromName("global"));
}

async function api(request: Request, env: Env, url: URL, body: string | null): Promise<ApiCall> {
  const cost = units(request.method, url.pathname, body);
  const keyed = await hasKey(request, env);
  let charge: Charge | null = null;
  if (!keyed) {
    const metered = await meter(request, env, cost);
    if (metered instanceof Response) return { response: withCors(metered), units: cost, keyed };
    charge = metered;
  }
  const http = {
    method: request.method,
    // pathname keeps percent escapes, so an escaped comma inside a label reaches the parser.
    path: url.pathname + url.search,
    headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
    body,
  };
  try {
    const job = await runJob(env, { http });
    const output = job.output;
    if (!isHttpOutput(output)) throw new JobError("GPU worker returned no HTTP response", 502);
    // The app failed on our side (its backend, not the request): the caller pays nothing.
    if (charge && output.status >= 500) await refund(charge);
    const response = new Response(output.body, {
      status: output.status,
      headers: { ...output.headers, ...charge?.headers },
    });
    return { response: withCors(response), units: cost, keyed, job };
  } catch (error) {
    if (!(error instanceof JobError)) throw error;
    if (charge) await refund(charge);
    return { response: withCors(json({ detail: error.message }, error.status)), units: cost, keyed };
  }
}

/**
 * One data point per API call in STATS, and one more call in today's LIVE counts, both shown at
 * /stats. It records how the API is used, never what is classified: no text, labels, or IP
 * address. The client ID only counts distinct users per day; it is keyed with a Worker secret and
 * the date, so it can't be reversed to an address or linked across days.
 *
 *   blobs:   endpoint, status, tier (free|key), source, client kind, country, other site's host
 *   doubles: units, Worker latency ms, RunPod queue and cold start ms, GPU ms
 */
async function record(request: Request, url: URL, call: ApiCall, ms: number, env: Env): Promise<void> {
  const origin = request.headers.get("Origin");
  const source = !origin ? "direct" : origin === url.origin ? "playground" : "website";
  // Without the secret (local dev) the call is still counted, just not per client.
  const day = new Date().toISOString().slice(0, 10);
  const client = env.RUNPOD_API_KEY ? await dailyClient(request, day, env.RUNPOD_API_KEY) : "none";
  const status = call.response.status;
  try {
    // Queues the point without waiting for storage.
    env.STATS?.writeDataPoint({
      blobs: [
        routeOf(request.method, url.pathname, url.search),
        String(status),
        call.keyed ? "key" : "free",
        source,
        clientKind(request.headers.get("User-Agent") ?? ""),
        String(request.cf?.country ?? ""),
        source === "website" ? hostOf(origin) : "",
      ],
      doubles: [call.units, ms, call.job?.delayTime ?? 0, call.job?.executionTime ?? 0],
      indexes: [client],
    });
  } catch (error) {
    // The live count still goes ahead.
    console.error("API call not recorded", error);
  }
  await liveStats(env).add({ units: call.units, status, keyed: call.keyed, client });
}

// The app's own fixed routes (FastAPI adds the docs). Anything else is named by its shape, so a
// caller can't put words on the public page by calling a made-up path or method.
const ROUTES = new Set(["/v1/score", "/v1/systemone", "/v1/models", "/docs", "/redoc", "/openapi.json"]);
const METHODS = new Set(["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]);

/** The route an API call used, without its labels or text. */
export function routeOf(method: string, path: string, search: string): string {
  if (!METHODS.has(method)) return "other";
  if (ROUTES.has(path)) return `${method} ${path}`;
  if (path === "/") return method === "POST" ? "POST /" : search ? "GET /?labels" : "GET /";
  return path.split("/").filter(Boolean).length >= 2 ? `${method} /<labels>/<text>` : "other";
}

/** A coarse client family from the User-Agent, for telling scripts from browsers. */
export function clientKind(userAgent: string): string {
  if (!userAgent) return "none";
  if (/^curl\//i.test(userAgent)) return "curl";
  if (/typesafe/i.test(userAgent)) return "typesafe-sdk";
  if (/python|httpx|aiohttp|urllib/i.test(userAgent)) return "python";
  if (/node|undici|axios/i.test(userAgent)) return "node";
  if (/bot|crawler|spider/i.test(userAgent)) return "bot";
  if (/mozilla/i.test(userAgent)) return "browser";
  return "other";
}

function hostOf(origin: string | null): string {
  try {
    return new URL(origin ?? "").host;
  } catch {
    return "invalid";
  }
}

async function dailyClient(request: Request, day: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const message = `stats-v1:${day}:${ip.includes(":") ? ipv6Prefix(ip) : ip}`;
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message));
  // 16 bytes is plenty to count distinct clients, and fits the 96-byte index limit.
  return Array.from(new Uint8Array(mac).slice(0, 16), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Units a request costs: one, or one per text in a batch or per question in System One. */
export function units(method: string, path: string, body: string | null): number {
  if (method !== "POST" || !body) return 1;
  try {
    const data = JSON.parse(body);
    if (path === "/" && Array.isArray(data?.input)) return Math.max(1, data.input.length);
    if (path === "/v1/systemone" && data?.questions && typeof data.questions === "object") {
      return Math.max(1, Object.keys(data.questions).length);
    }
  } catch {
    // Malformed bodies cost one unit; the app rejects them.
  }
  return 1;
}

async function meter(request: Request, env: Env, cost: number): Promise<Charge | Response> {
  const limit = Number(env.DAILY_LIMIT ?? 1000);
  const globalLimit = Number(env.GLOBAL_DAILY_LIMIT ?? 20000);
  const now = new Date();
  const day = now.toISOString().slice(0, 10);
  const resetAt = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
  const retryAfter = String(Math.ceil((resetAt - now.getTime()) / 1000));
  const client = env.QUOTA.get(env.QUOTA.idFromName(`ip:${await clientId(request)}`));
  const global = env.QUOTA.get(env.QUOTA.idFromName("global"));

  const mine = await client.take(cost, limit, day, resetAt);
  const headers = {
    "X-RateLimit-Limit": String(limit),
    "X-RateLimit-Remaining": String(mine.remaining),
    "X-RateLimit-Reset": retryAfter,
  };
  if (!mine.allowed) {
    const detail =
      cost > 1
        ? `This request needs ${cost} units; ${mine.remaining} of your ${limit} free units are left today.`
        : `Free limit of ${limit} requests per day reached for your IP.`;
    return json({ detail: `${detail} It resets at 00:00 UTC.` }, 429, {
      ...headers,
      "Retry-After": retryAfter,
    });
  }
  if (!(await global.take(cost, globalLimit, day, resetAt)).allowed) {
    await client.refund(cost, day);
    return json({ detail: "The free tier is at capacity for today. It resets at 00:00 UTC." }, 429, {
      "Retry-After": retryAfter,
    });
  }
  return { units: cost, day, client, global, headers };
}

async function refund(charge: Charge): Promise<void> {
  await Promise.all([
    charge.client.refund(charge.units, charge.day),
    charge.global.refund(charge.units, charge.day),
  ]);
}

/** A hashed client key: the IPv4 address, or the /64 an IPv6 address sits in. */
export async function clientId(request: Request): Promise<string> {
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const hash = await sha256(`quota-v1:${ip.includes(":") ? ipv6Prefix(ip) : ip}`);
  return Array.from(hash, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The first four groups of an IPv6 address: one household or server usually owns a /64. */
export function ipv6Prefix(ip: string): string {
  const [head, tail] = ip.toLowerCase().split("::");
  const left = head ? head.split(":") : [];
  const right = tail ? tail.split(":") : [];
  const groups =
    tail === undefined ? left : [...left, ...Array(8 - left.length - right.length).fill("0"), ...right];
  return groups
    .slice(0, 4)
    .map((g) => g.replace(/^0+(?=.)/, ""))
    .join(":");
}

function endpoint(env: Env): string {
  return `${env.RUNPOD_API_BASE ?? "https://api.runpod.ai/v2"}/${env.RUNPOD_ENDPOINT_ID}`;
}

async function runJob(env: Env, input: unknown): Promise<Job> {
  const base = endpoint(env);
  const headers = {
    Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    "Content-Type": "application/json",
  };
  const timeout = Number(env.JOB_TIMEOUT_SECONDS ?? 240) * 1000;
  const poll = Number(env.POLL_SECONDS ?? 5) * 1000;
  const deadline = Date.now() + timeout;
  let job = await runpod(
    fetch(`${base}/runsync`, { method: "POST", headers, body: JSON.stringify({ input }) }),
  );
  while (PENDING.has(job.status ?? "")) {
    if (Date.now() + poll >= deadline) {
      // Best effort, so an abandoned job does not keep a GPU busy.
      await fetch(`${base}/cancel/${job.id}`, { method: "POST", headers }).catch(() => undefined);
      console.error("RunPod job timed out", job.id, job.status);
      throw new JobError("The GPU took too long to answer. Please try again.", 504);
    }
    await new Promise((resolve) => setTimeout(resolve, poll));
    job = await runpod(fetch(`${base}/status/${job.id}`, { headers }));
  }
  if (job.status !== "COMPLETED") {
    // The error can carry a worker traceback; log it, do not return it to callers.
    console.error("RunPod job failed", job.id, job.status, job.error);
    throw new JobError(UNAVAILABLE, 502);
  }
  return job;
}

async function runpod(pending: Promise<Response>): Promise<Job> {
  let response: Response;
  try {
    response = await pending;
  } catch (error) {
    console.error("RunPod unreachable", error);
    throw new JobError(UNAVAILABLE, 502);
  }
  if (!response.ok) {
    // 401 or 403 here means the Worker's RUNPOD_API_KEY secret is missing or wrong.
    console.error("RunPod answered", response.status);
    throw new JobError(UNAVAILABLE, 502);
  }
  try {
    return (await response.json()) as Job;
  } catch {
    console.error("RunPod returned invalid JSON");
    throw new JobError(UNAVAILABLE, 502);
  }
}

async function health(env: Env): Promise<Response> {
  const response = await fetch(`${endpoint(env)}/health`, {
    headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` },
  }).catch(() => null);
  if (!response?.ok) return json({ status: "degraded", runpod: response?.status ?? null }, 503);
  const body = (await response.json()) as { workers?: unknown; jobs?: unknown };
  return json({ status: "ready", workers: body.workers, jobs: body.jobs });
}

/**
 * Whether the caller sent one of API_KEYS. Any other key counts as the free tier rather than an
 * error, so clients that insist on a key (typesafe-sdk) work with a placeholder like "free".
 */
async function hasKey(request: Request, env: Env): Promise<boolean> {
  const header = request.headers.get("Authorization") ?? "";
  const space = header.indexOf(" ");
  if (space < 0 || header.slice(0, space).toLowerCase() !== "bearer") return false;
  const token = await sha256(header.slice(space + 1));
  const keys = (env.API_KEYS ?? "")
    .split(",")
    .map((key) => key.trim())
    .filter(Boolean);
  let match = false;
  // Compare digests against every key without stopping early, so timing reveals nothing.
  for (const key of keys) match = equal(token, await sha256(key)) || match;
  return match;
}

/**
 * Browsers, and link-preview bots, which often accept any type without naming HTML yet need the
 * page's title and preview tags. Only command-line tools not asking for HTML get the usage text.
 */
function wantsPage(request: Request): boolean {
  if ((request.headers.get("Accept") ?? "").includes("text/html")) return true;
  return !COMMAND_LINE.test(request.headers.get("User-Agent") ?? "");
}

async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

function equal(a: Uint8Array, b: Uint8Array): boolean {
  let difference = a.length ^ b.length;
  for (let i = 0; i < a.length; i++) difference |= a[i] ^ (b[i] ?? 0);
  return difference === 0;
}

function isHttpOutput(value: unknown): value is HttpOutput {
  const output = value as HttpOutput | null;
  return (
    typeof output?.status === "number" &&
    typeof output.body === "string" &&
    typeof output.headers === "object"
  );
}

function withCors(response: Response): Response {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(CORS)) headers.set(name, value);
  return new Response(response.body, { status: response.status, headers });
}

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}
