/**
 * Public API for syn on Cloudflare Workers.
 *
 * Holds no API logic. It checks the caller's bearer key, wraps the request as the RunPod job
 * input {"http": {method, path, headers, body}}, waits for the GPU worker (polling through
 * cold starts), and returns the response the Python app produced there. /health reports
 * RunPod worker counts without waking a GPU.
 */

export interface Env {
  /** Secret: the RunPod serverless endpoint running deploy/runpod/handler.py. */
  RUNPOD_ENDPOINT_ID: string;
  /** Secret: the RunPod API key. */
  RUNPOD_API_KEY: string;
  /** Secret: comma-separated keys callers may send as `Authorization: Bearer <key>`. */
  API_KEYS: string;
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

const PENDING = new Set(["IN_QUEUE", "IN_PROGRESS"]);

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
    if (url.pathname === "/health") return health(env);
    if (!(await authorized(request, env))) {
      return json({ detail: "Missing or invalid bearer token" }, 401, {
        "WWW-Authenticate": "Bearer",
      });
    }
    const http = {
      method: request.method,
      // pathname keeps percent escapes, so an escaped comma inside a label reaches the parser.
      path: url.pathname + url.search,
      headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
      body: request.method === "GET" || request.method === "HEAD" ? null : await request.text(),
    };
    try {
      const output = await runJob(env, { http });
      if (!isHttpOutput(output)) throw new JobError("GPU worker returned no HTTP response", 502);
      return new Response(output.body, { status: output.status, headers: output.headers });
    } catch (error) {
      if (error instanceof JobError) return json({ detail: error.message }, error.status);
      throw error;
    }
  },
};

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
      throw new JobError(`GPU worker did not finish within ${timeout / 1000}s`, 504);
    }
    await new Promise((resolve) => setTimeout(resolve, poll));
    job = await runpod(fetch(`${base}/status/${job.id}`, { headers }));
  }
  if (job.status !== "COMPLETED") {
    // The error can carry a worker traceback; log it, do not return it to callers.
    console.error("RunPod job failed", job.id, job.status, job.error);
    throw new JobError(`Scoring job ${job.status ?? "returned no status"}`, 502);
  }
  return job.output;
}

async function runpod(pending: Promise<Response>): Promise<Job> {
  let response: Response;
  try {
    response = await pending;
  } catch (error) {
    throw new JobError(`RunPod unreachable: ${error}`, 502);
  }
  if (!response.ok) throw new JobError(`RunPod answered ${response.status}`, 502);
  try {
    return (await response.json()) as Job;
  } catch {
    throw new JobError("RunPod returned invalid JSON", 502);
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

async function authorized(request: Request, env: Env): Promise<boolean> {
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

function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}
