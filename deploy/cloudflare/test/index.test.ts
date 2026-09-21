import { afterEach, describe, expect, it, vi } from "vitest";
import worker, { type Env, ipv6Prefix, Quota, units } from "../src/index";

const RUNPOD = "https://api.runpod.ai/v2/ep1";
const OK = { status: 200, headers: { "content-type": "text/plain" }, body: "spam\n" };
const DONE = { id: "j", status: "COMPLETED", output: OK };

function storage() {
  const data = new Map<string, unknown>();
  let alarm: number | null = null;
  return {
    get: async (key: string) => data.get(key),
    put: async (key: string, value: unknown) => void data.set(key, value),
    deleteAll: async () => void data.clear(),
    getAlarm: async () => alarm,
    setAlarm: async (at: number) => void (alarm = at),
  };
}

function quotaNamespace() {
  const objects = new Map<string, Quota>();
  return {
    idFromName: (name: string) => name,
    get(id: string) {
      if (!objects.has(id)) {
        objects.set(id, new Quota({ storage: storage() } as unknown as DurableObjectState, {}));
      }
      return objects.get(id)!;
    },
  };
}

function makeEnv(overrides: Partial<Env> = {}): Env {
  return {
    RUNPOD_ENDPOINT_ID: "ep1",
    RUNPOD_API_KEY: "rp-secret",
    API_KEYS: "key-a, key-b",
    QUOTA: quotaNamespace() as unknown as Env["QUOTA"],
    ASSETS: { fetch: async () => new Response("<html>page</html>") } as unknown as Fetcher,
    DAILY_LIMIT: "3",
    GLOBAL_DAILY_LIMIT: "100",
    JOB_TIMEOUT_SECONDS: "60",
    POLL_SECONDS: "0",
    ...overrides,
  };
}

/** Stub RunPod: answers fetches in order and records each call. */
function runpod(...replies: (object | Response)[]) {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      const reply = replies.shift() ?? {};
      return reply instanceof Response ? reply : Response.json(reply);
    }),
  );
  return calls;
}

function call(env: Env, path: string, init: RequestInit & { ip?: string; key?: string } = {}) {
  const headers = new Headers(init.headers);
  headers.set("CF-Connecting-IP", init.ip ?? "203.0.113.7");
  if (init.key) headers.set("Authorization", `Bearer ${init.key}`);
  return worker.fetch(new Request(`https://sifty.example${path}`, { ...init, headers }), env);
}

async function detail(response: Response): Promise<string> {
  return ((await response.json()) as { detail: string }).detail;
}

afterEach(() => vi.unstubAllGlobals());

describe("free tier", () => {
  it("serves anonymous requests and counts them per IP", async () => {
    const env = makeEnv();
    runpod(DONE, DONE, DONE, DONE);
    const first = await call(env, "/a,b/hi");
    expect(first.status).toBe(200);
    expect(first.headers.get("X-RateLimit-Limit")).toBe("3");
    expect(first.headers.get("X-RateLimit-Remaining")).toBe("2");
    expect(Number(first.headers.get("X-RateLimit-Reset"))).toBeGreaterThan(0);
    await call(env, "/a,b/hi");
    await call(env, "/a,b/hi");
    const calls = runpod();
    const over = await call(env, "/a,b/hi");
    expect(over.status).toBe(429);
    expect(Number(over.headers.get("Retry-After"))).toBeGreaterThan(0);
    expect(await detail(over)).toContain("resets at 00:00 UTC");
    expect(calls).toHaveLength(0);
    // Another address has its own allowance.
    runpod(DONE);
    expect((await call(env, "/a,b/hi", { ip: "198.51.100.9" })).status).toBe(200);
  });

  it("lets configured keys skip the limits, and treats any other key as the free tier", async () => {
    const env = makeEnv({ DAILY_LIMIT: "1" });
    runpod(DONE, DONE, DONE);
    for (let i = 0; i < 2; i++) {
      const keyed = await call(env, "/a,b/hi", { key: "key-b" });
      expect(keyed.status).toBe(200);
      expect(keyed.headers.get("X-RateLimit-Remaining")).toBeNull();
    }
    expect((await call(env, "/a,b/hi", { key: "free" })).status).toBe(200);
    expect((await call(env, "/a,b/hi", { key: "free" })).status).toBe(429);
  });

  it("charges a batch per text and System One per question, and a batch must fit", async () => {
    expect(units("POST", "/", JSON.stringify({ input: ["a", "b", "c"], labels: [] }))).toBe(3);
    expect(units("POST", "/", JSON.stringify({ input: "a", labels: [] }))).toBe(1);
    expect(units("POST", "/v1/systemone", JSON.stringify({ questions: { a: {}, b: {} } }))).toBe(2);
    expect(units("POST", "/", "not json")).toBe(1);
    expect(units("GET", "/a,b/hi", null)).toBe(1);

    const env = makeEnv({ DAILY_LIMIT: "2" });
    const calls = runpod();
    const batch = await call(env, "/", {
      method: "POST",
      body: JSON.stringify({ input: ["a", "b", "c"], labels: ["x", "y"] }),
    });
    expect(batch.status).toBe(429);
    expect(await detail(batch)).toContain("needs 3 units");
    expect(calls).toHaveLength(0);
  });

  it("stops everyone at the global cap without charging the refused client", async () => {
    const env = makeEnv({ GLOBAL_DAILY_LIMIT: "1" });
    runpod(DONE);
    expect((await call(env, "/a,b/hi", { ip: "192.0.2.1" })).status).toBe(200);
    const refused = await call(env, "/a,b/hi", { ip: "192.0.2.2" });
    expect(refused.status).toBe(429);
    expect(await detail(refused)).toContain("at capacity");
    // With room again, the refused client still has its full allowance: the refusal was free.
    runpod(DONE);
    const later = await call({ ...env, GLOBAL_DAILY_LIMIT: "100" }, "/a,b/hi", { ip: "192.0.2.2" });
    expect(later.headers.get("X-RateLimit-Remaining")).toBe("2");
  });

  it("refunds units when the GPU job fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const env = makeEnv();
    runpod({ id: "j", status: "FAILED", error: "boom" }, DONE);
    expect((await call(env, "/a,b/hi")).status).toBe(502);
    const next = await call(env, "/a,b/hi");
    expect(next.headers.get("X-RateLimit-Remaining")).toBe("2");
  });

  it("refunds units when the app reports a server error, but not for a bad request", async () => {
    const env = makeEnv();
    const failed = { status: 502, headers: {}, body: '{"detail":"Scoring backend failed"}' };
    const rejected = { status: 422, headers: {}, body: '{"detail":"bad"}' };
    runpod({ id: "j", status: "COMPLETED", output: failed }, DONE, { id: "j", status: "COMPLETED", output: rejected }, DONE);
    expect((await call(env, "/a,b/hi")).status).toBe(502);
    expect((await call(env, "/a,b/hi")).headers.get("X-RateLimit-Remaining")).toBe("2");
    expect((await call(env, "/a/hi")).status).toBe(422);
    expect((await call(env, "/a,b/hi")).headers.get("X-RateLimit-Remaining")).toBe("0");
  });

  it("groups IPv6 clients by /64", async () => {
    expect(ipv6Prefix("2001:db8:abcd:12:1:2:3:4")).toBe("2001:db8:abcd:12");
    expect(ipv6Prefix("2001:db8::1")).toBe("2001:db8:0:0");
    expect(ipv6Prefix("2001:0DB8:0000:0012::9")).toBe("2001:db8:0:12");
    const env = makeEnv({ DAILY_LIMIT: "1" });
    runpod(DONE, DONE);
    expect((await call(env, "/a,b/hi", { ip: "2001:db8:abcd:12::1" })).status).toBe(200);
    expect((await call(env, "/a,b/hi", { ip: "2001:db8:abcd:12::ffff" })).status).toBe(429);
    expect((await call(env, "/a,b/hi", { ip: "2001:db8:abcd:13::1" })).status).toBe(200);
  });
});

describe("quota object", () => {
  it("resets on a new day and forgets everything at the alarm", async () => {
    const quota = new Quota({ storage: storage() } as unknown as DurableObjectState, {});
    expect(await quota.take(2, 3, "2026-09-21", 1)).toEqual({ allowed: true, remaining: 1 });
    expect(await quota.take(2, 3, "2026-09-21", 1)).toEqual({ allowed: false, remaining: 1 });
    expect(await quota.take(2, 3, "2026-09-22", 1)).toEqual({ allowed: true, remaining: 1 });
    await quota.refund(2, "2026-09-22");
    expect(await quota.take(3, 3, "2026-09-22", 1)).toEqual({ allowed: true, remaining: 0 });
    await quota.alarm();
    expect(await quota.take(3, 3, "2026-09-22", 1)).toEqual({ allowed: true, remaining: 0 });
  });
});

describe("page and CORS", () => {
  it("serves the page to browsers asking for / and the API to everything else", async () => {
    const env = makeEnv();
    const page = await call(env, "/", { headers: { Accept: "text/html,application/xhtml+xml" } });
    expect(await page.text()).toBe("<html>page</html>");
    const calls = runpod(DONE);
    await call(env, "/?labels=a,b&text=hi", { headers: { Accept: "text/html" } });
    expect(calls).toHaveLength(1);
  });

  it("answers preflight and marks API responses cross-origin", async () => {
    const env = makeEnv();
    const preflight = await call(env, "/", { method: "OPTIONS" });
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get("Access-Control-Allow-Headers")).toContain("Authorization");
    runpod(DONE);
    const response = await call(env, "/a,b/hi");
    expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(response.headers.get("Access-Control-Expose-Headers")).toContain("X-RateLimit-Remaining");
  });
});

describe("proxy", () => {
  it("wraps the raw request as a RunPod job and returns the app's response", async () => {
    const calls = runpod({
      id: "j",
      status: "COMPLETED",
      output: {
        status: 200,
        headers: { "content-type": "text/plain", "x-syn-selected": "spam" },
        body: "spam\n",
      },
    });
    const response = await call(makeEnv(), "/spam,a%2Cb/Win+a+free+iPhone?verbose=1");
    expect(response.status).toBe(200);
    expect(await response.text()).toBe("spam\n");
    expect(response.headers.get("x-syn-selected")).toBe("spam");
    const [{ url, init }] = calls;
    expect(url).toBe(`${RUNPOD}/runsync`);
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer rp-secret");
    expect(JSON.parse(init?.body as string)).toEqual({
      input: {
        http: {
          method: "GET",
          path: "/spam,a%2Cb/Win+a+free+iPhone?verbose=1",
          headers: { "content-type": "application/json" },
          body: null,
        },
      },
    });
  });

  it("forwards POST bodies and passes the app's error status through", async () => {
    const calls = runpod({
      id: "j",
      status: "COMPLETED",
      output: {
        status: 422,
        headers: { "content-type": "application/json" },
        body: '{"detail":"bad"}',
      },
    });
    const body = JSON.stringify({ state: "hi", model: "m", questions: {} });
    const response = await call(makeEnv(), "/v1/systemone", {
      method: "POST",
      body,
      headers: { "content-type": "application/json" },
    });
    expect(response.status).toBe(422);
    expect(await response.json()).toEqual({ detail: "bad" });
    expect(JSON.parse(calls[0].init?.body as string).input.http).toMatchObject({
      method: "POST",
      path: "/v1/systemone",
      body,
    });
  });

  it("polls through a cold start", async () => {
    const calls = runpod(
      { id: "j9", status: "IN_QUEUE" },
      { id: "j9", status: "IN_PROGRESS" },
      DONE,
    );
    expect(await (await call(makeEnv(), "/a,b/hi")).text()).toBe("spam\n");
    expect(calls.map((c) => c.url)).toEqual([
      `${RUNPOD}/runsync`,
      `${RUNPOD}/status/j9`,
      `${RUNPOD}/status/j9`,
    ]);
  });

  it("cancels the job and answers 504 at the deadline", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const calls = runpod({ id: "j9", status: "IN_QUEUE" }, {});
    const response = await call(makeEnv({ JOB_TIMEOUT_SECONDS: "0" }), "/a,b/hi");
    expect(response.status).toBe(504);
    expect(calls.at(-1)).toMatchObject({ url: `${RUNPOD}/cancel/j9` });
  });

  it("answers 502 with a plain message and keeps the internals in the logs", async () => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const env = makeEnv({ DAILY_LIMIT: "100" });
    runpod({ id: "j", status: "FAILED", error: "Traceback: /secret/path.py" });
    const failed = await call(env, "/a,b/hi");
    expect(failed.status).toBe(502);
    expect(await failed.text()).not.toContain("secret");

    // A missing or wrong RunPod key: callers see no internals, the log names the cause.
    runpod(new Response("nope", { status: 401 }));
    const unauthorized = await detail(await call(env, "/a,b/hi"));
    expect(unauthorized).toContain("temporarily unavailable");
    expect(unauthorized).not.toMatch(/RunPod|401/);
    expect(logged).toHaveBeenCalledWith("RunPod answered", 401);

    runpod({ id: "j", status: "COMPLETED", output: { unexpected: true } });
    expect((await call(env, "/a,b/hi")).status).toBe(502);
  });
});

describe("health", () => {
  it("reports RunPod worker counts without a key, a job, or a charge", async () => {
    const calls = runpod({ workers: { idle: 1, running: 0 }, jobs: { inQueue: 0 } });
    const response = await call(makeEnv(), "/health");
    expect(await response.json()).toEqual({
      status: "ready",
      workers: { idle: 1, running: 0 },
      jobs: { inQueue: 0 },
    });
    expect(response.headers.get("X-RateLimit-Remaining")).toBeNull();
    expect(calls.map((c) => c.url)).toEqual([`${RUNPOD}/health`]);
  });

  it("is degraded when RunPod does not answer", async () => {
    runpod(new Response("down", { status: 500 }));
    expect((await call(makeEnv(), "/health")).status).toBe(503);
  });
});
