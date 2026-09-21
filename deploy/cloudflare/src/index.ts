/**
 * Public API and web page for syn on Cloudflare Workers.
 *
 * Holds no API logic. It meters the request, wraps it as the RunPod job input
 * {"http": {method, path, headers, body}}, waits for the GPU worker (polling through cold
 * starts), and returns the response the Python app produced there.
 *
 * Free for everyone: requests without a known key are counted per client IP per UTC day, and
 * against a global daily cap that bounds GPU spend. Keys listed in API_KEYS skip both. Browsers
 * asking for `/` get the page in public/. /health reports RunPod worker counts without waking a
 * GPU.
 */

import type { Quota } from "./quota";

export { Quota } from "./quota";

export interface Env {
  /** Secret: the RunPod serverless endpoint running deploy/runpod/handler.py. */
  RUNPOD_ENDPOINT_ID: string;
  /** Secret: the RunPod API key. */
  RUNPOD_API_KEY: string;
  /** Secret: comma-separated keys that skip the free-tier limits. */
  API_KEYS?: string;
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
}

interface Job {
  id?: string;
  status?: string;
  output?: unknown;
  error?: unknown;
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

const PENDING = new Set(["IN_QUEUE", "IN_PROGRESS"]);
// What callers see when the GPU side fails; the specifics go to the Worker's logs.
const UNAVAILABLE = "The scoring service is temporarily unavailable. Please try again shortly.";
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
  async fetch(request: Request, env: Env): Promise<Response> {
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
    if (request.method === "GET" && url.pathname === "/" && !url.search && wantsHtml(request)) {
      return env.ASSETS.fetch(request);
    }
    const body =
      request.method === "GET" || request.method === "HEAD" ? null : await request.text();
    let charge: Charge | null = null;
    if (!(await hasKey(request, env))) {
      const metered = await meter(request, env, units(request.method, url.pathname, body));
      if (metered instanceof Response) return withCors(metered);
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
      const output = await runJob(env, { http });
      if (!isHttpOutput(output)) throw new JobError("GPU worker returned no HTTP response", 502);
      // The app failed on our side (its backend, not the request): the caller pays nothing.
      if (charge && output.status >= 500) await refund(charge);
      return withCors(
        new Response(output.body, {
          status: output.status,
          headers: { ...output.headers, ...charge?.headers },
        }),
      );
    } catch (error) {
      if (!(error instanceof JobError)) throw error;
      if (charge) await refund(charge);
      return withCors(json({ detail: error.message }, error.status));
    }
  },
};

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

async function runJob(env: Env, input: unknown): Promise<unknown> {
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
  return job.output;
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

function wantsHtml(request: Request): boolean {
  return (request.headers.get("Accept") ?? "").includes("text/html");
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
