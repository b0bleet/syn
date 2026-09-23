import { afterEach, describe, expect, it, vi } from "vitest";
import worker, {
  clientKind,
  type Env,
  ipv6Prefix,
  LiveStats,
  Quota,
  routeOf,
  STATIC_FILES,
  units,
} from "../src/index";

const RUNPOD = "https://api.runpod.ai/v2/ep1";
const OK = { status: 200, headers: { "content-type": "text/plain" }, body: "spam\n" };
const DONE = { id: "j", status: "COMPLETED", output: OK };

/** Durable Object storage in memory. Values are copied in and out, as workerd does. */
function storage() {
  const data = new Map<string, unknown>();
  let alarm: number | null = null;
  const store = {
    data,
    writes: 0,
    get: async (key: string) => structuredClone(data.get(key)),
    put: async (key: string, value: unknown) => {
      store.writes += 1;
      data.set(key, structuredClone(value));
    },
    delete: async (keys: string | string[]) => {
      for (const key of [keys].flat()) data.delete(key);
    },
    deleteAll: async () => void data.clear(),
    getAlarm: async () => alarm,
    setAlarm: async (at: number) => void (alarm = at),
    /** The runtime consumes an alarm as it fires. */
    deleteAlarm: async () => void (alarm = null),
  };
  return store;
}

/** Durable Objects by name, each with its own storage. */
function namespace<T>(make: (state: DurableObjectState) => T) {
  const objects = new Map<string, T>();
  return {
    idFromName: (name: string) => name,
    get(id: string) {
      if (!objects.has(id)) objects.set(id, make({ storage: storage() } as unknown as DurableObjectState));
      return objects.get(id)!;
    },
  };
}

function makeEnv(overrides: Partial<Env> = {}): Env {
  const env = {
    RUNPOD_ENDPOINT_ID: "ep1",
    RUNPOD_API_KEY: "rp-secret",
    API_KEYS: "key-a, key-b",
    QUOTA: namespace((state) => new Quota(state, {})) as unknown as Env["QUOTA"],
    ASSETS: { fetch: async () => new Response("<html>page</html>") } as unknown as Fetcher,
    DAILY_LIMIT: "3",
    GLOBAL_DAILY_LIMIT: "100",
    JOB_TIMEOUT_SECONDS: "60",
    POLL_SECONDS: "0",
    ...overrides,
  } as Env;
  env.LIVE ??= namespace((state) => new LiveStats(state, env)) as unknown as Env["LIVE"];
  return env;
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

async function call(
  env: Env, path: string,
  init: RequestInit & { ip?: string; key?: string; cf?: Partial<IncomingRequestCfProperties> } = {},
) {
  const headers = new Headers(init.headers);
  headers.set("CF-Connecting-IP", init.ip ?? "203.0.113.7");
  if (init.key) headers.set("Authorization", `Bearer ${init.key}`);
  // What the Worker finishes after answering, such as recording the call, is done before a test looks.
  const pending: Promise<unknown>[] = [];
  const ctx = { waitUntil: (promise: Promise<unknown>) => void pending.push(promise) };
  const request = new Request(`https://sifty.example${path}`, { ...init, headers });
  if (init.cf) Object.defineProperty(request, "cf", { value: init.cf });
  const response = await worker.fetch(request, env, ctx as unknown as ExecutionContext);
  await Promise.all(pending);
  return response;
}

async function detail(response: Response): Promise<string> {
  return ((await response.json()) as { detail: string }).detail;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

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

describe("contact form", () => {
  const message = { name: "Ada Lovelace", email: "ada@example.com", message: "Hello sifty!", company: "" };

  it("relays a valid message through Resend with the visitor as reply-to", async () => {
    const sends = vi.fn(async (_url: string, _init?: RequestInit) => Response.json({ id: "email-1" }));
    vi.stubGlobal("fetch", sends);
    const response = await call(makeEnv({ RESEND_API_KEY: "re-secret" }), "/contact", {
      method: "POST",
      headers: { "content-type": "application/json", Origin: "https://sifty.example" },
      body: JSON.stringify(message),
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: true });
    expect(sends).toHaveBeenCalledTimes(1);
    const [url, init] = sends.mock.calls[0];
    expect(url).toBe("https://api.resend.com/emails");
    expect(new Headers(init?.headers).get("authorization")).toBe("Bearer re-secret");
    expect(new Headers(init?.headers).get("idempotency-key")).toMatch(/^contact\/[0-9a-f]{64}$/);
    expect(JSON.parse(String(init?.body))).toEqual({
      from: "sifty <contact@send.sifty.dev>",
      to: ["emin@jolo.build"],
      reply_to: "ada@example.com",
      subject: "sifty contact from Ada Lovelace",
      text: "Name: Ada Lovelace\nEmail: ada@example.com\n\nHello sifty!",
    });
  });

  it("rejects invalid and cross-site submissions before sending", async () => {
    const sends = vi.fn();
    vi.stubGlobal("fetch", sends);
    const env = makeEnv({ RESEND_API_KEY: "re-secret" });
    const wrongOrigin = await call(env, "/contact", {
      method: "POST",
      headers: { "content-type": "application/json", Origin: "https://attacker.example" },
      body: JSON.stringify(message),
    });
    expect(wrongOrigin.status).toBe(403);
    const wrongType = await call(env, "/contact", { method: "POST", body: "hello" });
    expect(wrongType.status).toBe(415);
    const wrongEmail = await call(env, "/contact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...message, email: "not-an-email" }),
    });
    expect(wrongEmail.status).toBe(400);
    expect(sends).not.toHaveBeenCalled();
  });

  it("silently accepts the honeypot and reports missing or failed email service safely", async () => {
    const sends = vi.fn(async () => new Response("private provider error", { status: 403 }));
    vi.stubGlobal("fetch", sends);
    const trapped = await call(makeEnv(), "/contact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...message, company: "spam.example" }),
    });
    expect(await trapped.json()).toEqual({ ok: true });
    expect(sends).not.toHaveBeenCalled();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const missing = await call(makeEnv(), "/contact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(message),
    });
    expect(missing.status).toBe(503);
    const failed = await call(makeEnv({ RESEND_API_KEY: "re-secret" }), "/contact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(message),
    });
    expect(failed.status).toBe(502);
    expect(await detail(failed)).not.toContain("private provider error");
  });

  it("limits one address to five outbound messages a day", async () => {
    const sends = vi.fn(async () => Response.json({ id: "email" }));
    vi.stubGlobal("fetch", sends);
    const env = makeEnv({ RESEND_API_KEY: "re-secret" });
    for (let i = 0; i < 5; i++) {
      const response = await call(env, "/contact", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...message, message: `Message ${i}` }),
      });
      expect(response.status).toBe(200);
    }
    const limited = await call(env, "/contact", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(message),
    });
    expect(limited.status).toBe(429);
    expect(limited.headers.get("Retry-After")).toBeTruthy();
    expect(sends).toHaveBeenCalledTimes(5);
  });
});

describe("page and CORS", () => {
  it("serves the page to browsers and preview bots, and the API to curl and queries", async () => {
    const env = makeEnv();
    const calls = runpod(DONE, DONE);
    const page = await call(env, "/", { headers: { Accept: "text/html,application/xhtml+xml" } });
    expect(await page.text()).toBe("<html>page</html>");
    // Preview bots often accept anything, yet they need the page's title and preview tags.
    const bot = { Accept: "*/*", "User-Agent": "facebookexternalhit/1.1" };
    expect(await (await call(env, "/", { headers: bot })).text()).toBe("<html>page</html>");
    expect((await call(env, "/", { method: "HEAD", headers: bot })).status).toBe(200);
    expect(calls).toHaveLength(0);
    // curl gets the app's usage text, and a query is always an API call.
    await call(env, "/", { headers: { Accept: "*/*", "User-Agent": "curl/8.7.1" } });
    await call(env, "/?labels=a,b&text=hi", { headers: { Accept: "text/html" } });
    expect(calls).toHaveLength(2);
  });

  it("serves the page's files without a GPU job or a charge", async () => {
    const served: string[] = [];
    const assets = async (request: Request) => {
      served.push(new URL(request.url).pathname);
      return new Response("file");
    };
    const env = makeEnv({ DAILY_LIMIT: "1", ASSETS: { fetch: assets } as unknown as Fetcher });
    const calls = runpod(DONE);
    const files = [...STATIC_FILES, "/contact", "/contact.html"];
    for (const path of files) {
      const response = await call(env, path);
      expect(await response.text()).toBe("file");
      expect(response.headers.get("X-RateLimit-Remaining")).toBeNull();
    }
    expect(served).toEqual(files);
    expect(calls).toHaveLength(0);
    // The one free unit is still there.
    expect((await call(env, "/a,b/hi")).status).toBe(200);
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

/** A stand-in Analytics Engine dataset that keeps what the Worker writes. */
function statsSink() {
  const points: AnalyticsEngineDataPoint[] = [];
  const dataset = { writeDataPoint: (point: AnalyticsEngineDataPoint) => void points.push(point) };
  return { points, dataset: dataset as unknown as AnalyticsEngineDataset };
}

/** A LIVE namespace whose object refuses everything, as when over the free plan's limits. */
function refusingLive(): Env["LIVE"] {
  const refuse = async () => {
    throw new Error("limit exceeded");
  };
  return { idFromName: (name: string) => name, get: () => ({ add: refuse, page: refuse }) } as unknown as Env["LIVE"];
}

describe("private request logs", () => {
  it("captures a payload before the backend responds, then links its completion", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    let finish!: (response: Response) => void;
    const backend = new Promise<Response>((resolve) => { finish = resolve; });
    let entered!: () => void;
    const submitted = new Promise<void>((resolve) => { entered = resolve; });
    vi.stubGlobal("fetch", vi.fn(() => { entered(); return backend; }));
    const payload = { input: ["queued first text", "queued second text"], labels: ["a", "b"] };
    const pending = call(makeEnv({ REQUEST_LOGS: "true" }), "/", {
      method: "POST", body: JSON.stringify(payload), cf: { country: "AZ" },
    });
    await submitted;
    expect(logged).toHaveBeenCalledTimes(1);
    const received = logged.mock.calls[0][0];
    expect(received).toMatchObject({
      event: "sifty.request", phase: "received", message: "Received POST /", level: "info",
      request_id: expect.any(String), request: payload, country: "AZ", response: null,
    });
    expect(received.status).toBeUndefined();
    finish(Response.json(DONE));
    expect((await pending).status).toBe(200);
    expect(logged).toHaveBeenCalledTimes(2);
    expect(logged.mock.calls[1][0]).toMatchObject({
      phase: "completed", request_id: received.request_id, timestamp: received.timestamp,
      message: "Completed POST / — HTTP 200", status: 200, request: payload, response: "spam",
    });
  });

  it("records content and country in private logs without leaking them to public statistics", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const { points, dataset } = statsSink();
    const env = makeEnv({ REQUEST_LOGS: "true", STATS: dataset });
    const output = {
      model: "test-model", results: [{ label: "billing", confidence: 0.9 }],
      debug_secret: "response-secret",
    };
    const calls = runpod({ ...DONE, delayTime: 11, executionTime: 22, output: {
      status: 200, headers: { "content-type": "application/json" }, body: JSON.stringify(output),
    } });
    const submitted = { input: ["private-ticket", "private-review"], labels: ["billing", "support"] };
    const body = JSON.stringify({ ...submitted, api_key: "body-secret" });
    const response = await call(env, "/?token=query-secret", {
      method: "POST", key: "key-a", body,
      headers: {
        Origin: "https://sifty.example", Cookie: "cookie-secret",
        "User-Agent": "python-httpx/0.28.1", "content-type": "application/json",
        "X-Api-Key": "header-secret",
      },
      cf: { country: "AZ", region: "Baku", city: "Baku", timezone: "Asia/Baku", asn: 123,
        asOrganization: "Example network", colo: "GYD" },
    });
    expect(await response.json()).toEqual(output);
    expect(JSON.parse(calls[0].init?.body as string).input.http.body).toBe(body);
    expect(logged).toHaveBeenCalledTimes(2);
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed")[0][0]).toMatchObject({
      event: "sifty.request", endpoint: "POST /", client: "python", status: 200,
      tier: "key", units: 2, country: "AZ", region: "Baku", city: "Baku", timezone: "Asia/Baku",
      asn: 123, network: "Example network", datacenter: "GYD", source: "playground",
      website: "sifty.example", request: submitted,
      response: { model: "test-model", results: output.results }, queue_ms: 11, gpu_ms: 22,
      timestamp: expect.any(String), duration_ms: expect.any(Number),
    });
    const privateData = JSON.stringify(logged.mock.calls);
    for (const secret of ["key-a", "rp-secret", "203.0.113.7", "cookie-secret", "header-secret", "body-secret", "query-secret", "response-secret"]) {
      expect(privateData).not.toContain(secret);
    }
    expect(JSON.stringify(points)).not.toContain("private-ticket");
    expect(JSON.stringify(points)).not.toContain("billing");
    cloudflareSql();
    const publicPage = await (await call(env, "/stats")).text();
    expect(publicPage).not.toContain("private-ticket");
    expect(publicPage).not.toContain("private-review");
    expect(logged).toHaveBeenCalledTimes(2);
  });

  it("decodes URL inputs without recording unrelated query parameters or referrer secrets", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const env = makeEnv({ REQUEST_LOGS: "true" });
    runpod(DONE, { ...DONE, output: { ...OK, body: "123\n" } });
    await call(env, "/spam,a%2Cb/Hello+%2B+world%2Fnext&part?q=Which+team%3F&api_key=hidden", {
      headers: { Referer: "https://user:password@example.org/private?secret=hidden" },
    });
    await call(env, "/?text=old&text=Win%2Ba+prize&labels=spam,not+spam&q=Is+this+spam%3F&token=hidden");
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed")[0][0]).toMatchObject({
      website: "example.org", request: { text: "Hello + world/next&part", labels: ["spam", "a,b"], question: "Which team?" },
      response: "spam",
    });
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed")[1][0]).toMatchObject({
      request: { text: "Win+a prize", labels: ["spam", "not spam"], question: "Is this spam?" }, response: "123",
    });
    expect(JSON.stringify(logged.mock.calls)).not.toMatch(/hidden|password|private\?/);
  });

  it.each([
    ["/v1/score", { context: "My card was charged twice", question: "Which team?", criteria: "Pick one", options: [{ id: "a", text: "Billing" }, { id: "b", text: "Support" }] }],
    ["/v1/systemone", { state: { ticket: "Charged twice" }, model: "syn-latest", questions: { billing: { type: "noul", instructions: "Is this billing?" } } }],
  ])("captures the supported content fields of %s", async (path, payload) => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    runpod(DONE);
    await call(makeEnv({ REQUEST_LOGS: "true" }), path, { method: "POST", body: JSON.stringify(payload) });
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed")[0][0].request).toEqual(payload);
  });

  it("logs quota refusals and backend failures, even when public statistics fail", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const env = makeEnv({ REQUEST_LOGS: "true", DAILY_LIMIT: "1", LIVE: refusingLive() });
    runpod({ status: "FAILED", error: "internal-backend-secret" }, DONE);
    await call(env, "/a,b/hi");
    await call(env, "/a,b/hi");
    await call(env, "/a,b/hi");
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed").map(([entry]) => entry.status)).toEqual([502, 200, 429]);
    expect(JSON.stringify(logged.mock.calls)).not.toContain("internal-backend-secret");
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed")[2][0].request.text).toBe("hi");
  });

  it("bounds large Unicode/escaped content without truncating the forwarded input or response", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const body = JSON.stringify({ input: '漢字"\\\n'.repeat(4000), labels: ["a", "b"] });
    const output = { ...OK, body: "漢".repeat(20_000) };
    const calls = runpod({ ...DONE, output });
    const response = await call(makeEnv({ REQUEST_LOGS: "true" }), "/", { method: "POST", body });
    const event = logged.mock.calls.filter(([entry]) => entry.phase === "completed")[0][0];
    expect(event.request.truncated).toBe(true);
    expect(event.response.truncated).toBe(true);
    expect(new TextEncoder().encode(JSON.stringify(event)).length).toBeLessThan(100_000);
    expect(JSON.parse(calls[0].init?.body as string).input.http.body).toBe(body);
    expect(await response.text()).toBe(output.body);
  });

  it("marks oversized and malformed API bodies, and excludes unrelated routes entirely", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const env = makeEnv({ REQUEST_LOGS: "true", DAILY_LIMIT: "10" });
    runpod(DONE, DONE, DONE, DONE);
    await call(env, "/", { method: "POST", body: JSON.stringify({ input: "x".repeat(130_000) }) });
    await call(env, "/", { method: "POST", body: "invalid-secret" });
    await call(env, "/unknown", { method: "POST", body: "unknown-secret" });
    await call(env, "/v1/models?token=query-secret");
    expect(logged.mock.calls.filter(([entry]) => entry.phase === "completed").map(([entry]) => entry.request)).toEqual([
      { omitted: "body_too_large", characters: expect.any(Number) }, { omitted: "invalid_json" },
    ]);
    expect(JSON.stringify(logged.mock.calls)).not.toMatch(/invalid-secret|unknown-secret|query-secret/);
  });

  it("does not save discovery, documentation, probes, or non-classification methods", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const env = makeEnv({ REQUEST_LOGS: "true" });
    const paths = ["/", "/?utm_source=website", "/docs", "/redoc", "/openapi.json", "/v1/models", "/unknown?text=probe&labels=a,b"];
    runpod(...Array(paths.length + 3).fill(DONE));
    for (const path of paths) await call(env, path, { key: "key-a", headers: { "User-Agent": "curl/8.7.1" } });
    await call(env, "/a,b/hi", { method: "HEAD", key: "key-a" });
    await call(env, "/a,b/hi", { method: "POST", body: "unknown-secret", key: "key-a" });
    await call(env, "/", { method: "PUT", body: "unknown-secret", key: "key-a" });
    expect(logged).not.toHaveBeenCalled();
  });

  it("supports disabling logs and excludes pages, stats, health, and contact submissions", async () => {
    const logged = vi.spyOn(console, "log").mockImplementation(() => undefined);
    runpod(DONE, DONE);
    await call(makeEnv(), "/a,b/hi");
    await call(makeEnv({ REQUEST_LOGS: "false" }), "/a,b/hi");
    const env = makeEnv({ REQUEST_LOGS: "true" });
    await call(env, "/");
    await call(env, "/robots.txt");
    await call(env, "/contact");
    await call(env, "/contact", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ company: "bot" }) });
    await call(env, "/", { method: "OPTIONS" });
    runpod({ workers: {} });
    await call(env, "/health");
    cloudflareSql();
    await call(env, "/stats");
    expect(logged).not.toHaveBeenCalled();
  });

  it("preserves API responses if logging fails", async () => {
    vi.spyOn(console, "log").mockImplementation(() => { throw new Error("logging unavailable"); });
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);
    runpod(DONE);
    const response = await call(makeEnv({ REQUEST_LOGS: "true" }), "/a,b/private-text");
    expect(response.status).toBe(200);
    expect(await response.text()).toBe("spam\n");
    expect(errors).toHaveBeenCalledWith("Private request log could not be written");
    expect(JSON.stringify(errors.mock.calls)).not.toContain("private-text");
  });
});

describe("API call statistics", () => {
  it("records each API call without its content, and nothing for the page or its files", async () => {
    const { points, dataset } = statsSink();
    const env = makeEnv({ STATS: dataset });
    runpod({ ...DONE, delayTime: 12000, executionTime: 800 }, DONE);
    await call(env, "/spam,ham/secret-text", { headers: { "User-Agent": "curl/8.7.1" } });
    await call(env, "/", {
      method: "POST",
      headers: { Origin: "https://sifty.example", "User-Agent": "Mozilla/5.0 (Macintosh)" },
      body: JSON.stringify({ input: ["secret-a", "secret-b"], labels: ["x", "y"] }),
      key: "key-a",
    });
    await call(env, "/", { headers: { Accept: "text/html" } });
    await call(env, "/robots.txt");
    await call(env, "/", { method: "OPTIONS" });

    expect(points).toHaveLength(2);
    const [get, post] = points;
    expect(get.blobs).toEqual(["GET /<labels>/<text>", "200", "free", "direct", "curl", "", ""]);
    expect(get.doubles?.[0]).toBe(1);
    expect(get.doubles?.slice(2)).toEqual([12000, 800]);
    expect(post.blobs).toEqual(["POST /", "200", "key", "playground", "browser", "", ""]);
    expect(post.doubles?.[0]).toBe(2);
    // Neither the text, the labels, nor the address is stored.
    const stored = JSON.stringify(points);
    for (const secret of ["secret", "spam,ham", "203.0.113.7"]) expect(stored).not.toContain(secret);
  });

  it("records refusals and failures, other sites, and a client ID that changes daily", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { points, dataset } = statsSink();
    const env = makeEnv({ STATS: dataset, DAILY_LIMIT: "1" });
    runpod({ id: "j", status: "FAILED", error: "boom" }, DONE, DONE);
    await call(env, "/a,b/hi", { headers: { Origin: "https://example.org" } });
    await call(env, "/a,b/hi");
    await call(env, "/a,b/hi");
    await call(env, "/a,b/hi", { ip: "198.51.100.9" });

    expect(points.map((p) => p.blobs?.[1])).toEqual(["502", "200", "429", "200"]);
    expect(points[0].blobs?.slice(3)).toEqual(["website", "none", "", "example.org"]);
    const ids = points.map((p) => p.indexes?.[0]);
    expect(ids[0]).toMatch(/^[0-9a-f]{32}$/);
    expect(new Set(ids.slice(0, 3)).size).toBe(1);
    expect(ids[3]).not.toBe(ids[0]);
    // The same client gets a new ID tomorrow, so days can't be linked.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 86_400_000);
    runpod(DONE);
    await call(env, "/a,b/hi");
    expect(points[4].indexes?.[0]).not.toBe(ids[0]);
  });

  it("still answers when recording fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const broken = {
      writeDataPoint: () => {
        throw new Error("dataset unavailable");
      },
    } as unknown as AnalyticsEngineDataset;
    runpod(DONE);
    expect((await call(makeEnv({ STATS: broken }), "/a,b/hi")).status).toBe(200);
    // Nor when the live counts fail, say over the free plan's Durable Object limits.
    runpod(DONE);
    expect((await call(makeEnv({ LIVE: refusingLive() }), "/a,b/hi")).status).toBe(200);
    // Without the secret that keys client IDs, as in local dev, the call is counted anyway.
    const { points, dataset } = statsSink();
    runpod(DONE);
    await call(makeEnv({ STATS: dataset, RUNPOD_API_KEY: "" }), "/a,b/hi");
    expect(points.map((p) => p.indexes)).toEqual([["none"]]);
  });

  it("names routes without their content, and client families", () => {
    expect(routeOf("GET", "/spam,ham/Win+a+prize", "")).toBe("GET /<labels>/<text>");
    expect(routeOf("GET", "/", "?labels=a,b&text=hi")).toBe("GET /?labels");
    expect(routeOf("GET", "/", "")).toBe("GET /");
    expect(routeOf("POST", "/", "")).toBe("POST /");
    expect(routeOf("POST", "/v1/systemone", "")).toBe("POST /v1/systemone");
    expect(routeOf("GET", "/v1/score", "?labels=a,b&text=hi")).toBe("GET /v1/score");
    expect(routeOf("GET", "/openapi.json", "")).toBe("GET /openapi.json");
    expect(routeOf("GET", "/unknown", "")).toBe("other");
    // Made-up paths and methods are named by shape, so nobody can write on the public page.
    expect(routeOf("GET", "/v1/visit-my-site.example", "")).toBe("GET /<labels>/<text>");
    expect(routeOf("GET", "/v1/buy", "")).toBe("GET /<labels>/<text>");
    expect(routeOf("SPAMSPAM", "/a,b/hi", "")).toBe("other");
    expect(clientKind("curl/8.7.1")).toBe("curl");
    expect(clientKind("typesafe-sdk/0.7.0")).toBe("typesafe-sdk");
    expect(clientKind("python-httpx/0.28.1")).toBe("python");
    expect(clientKind("Mozilla/5.0 (compatible; Googlebot/2.1)")).toBe("bot");
    expect(clientKind("Mozilla/5.0 (Macintosh) AppleWebKit Safari/605.1.15")).toBe("browser");
    expect(clientKind("")).toBe("none");
  });
});

/** Stub Cloudflare's SQL API: answers each statistics query by the columns it selects. */
function cloudflareSql(ok = true) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const pick = (sql: string) => {
    if (sql.includes("AS day")) return [{ day: "2026-09-21 00:00:00", calls: "40", users: "9" }];
    if (sql.includes("AS hour")) return [{ hour: "2026-09-22 00:00:00", calls: "6" }];
    if (sql.includes("AS endpoint")) return [{ endpoint: "POST /", calls: "30", texts: "45" }];
    if (sql.includes("AS source")) return [{ source: "direct", client: "<script>x</script>", calls: "2" }];
    if (sql.includes("gpu_seconds")) return [{ calls: "38", gpu_seconds: 19.4 }];
    if (sql.includes("AS status") || sql.includes("AS country")) return [];
    return [{ calls: "1234", texts: "1500", keyed: "4", limited: "7", failed: "0", p50_ms: 1302.6 }];
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      if (!ok) return new Response("account acc1: bad token", { status: 403 });
      return Response.json({ meta: [], data: pick(String(init?.body)) });
    }),
  );
  return calls;
}

/** Stub the Workers cache as a map from URL to response. */
function workersCache() {
  const store = new Map<string, Response>();
  vi.stubGlobal("caches", {
    default: {
      match: async (key: Request) => store.get(key.url)?.clone(),
      put: async (key: Request, response: Response) => void store.set(key.url, response),
    },
  });
  return store;
}

describe("public statistics page", () => {
  const connected = { STATS_ACCOUNT_ID: "acc1", STATS_API_TOKEN: "cf-read" };

  it("is open to everyone, and viewing it is neither charged nor recorded", async () => {
    const { points, dataset } = statsSink();
    const env = makeEnv({ ...connected, STATS: dataset, DAILY_LIMIT: "1" });
    const calls = cloudflareSql();
    const page = await call(env, "/stats");
    expect(page.status).toBe(200);
    expect(page.headers.get("content-type")).toContain("text/html");
    expect(await page.text()).toContain("<b>0</b>calls");
    expect(calls.length).toBe(8);
    expect(points).toHaveLength(0);
    runpod(DONE);
    expect((await call(env, "/a,b/hi")).status).toBe(200);
  });

  it("counts today's calls live, ahead of the tables", async () => {
    // Client IDs change daily; a fixed day keeps the users estimate the same on every run.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-09-21T12:00:00Z"));
    const env = makeEnv({ ...connected, DAILY_LIMIT: "1" });
    runpod(DONE, DONE, DONE);
    await call(env, "/a,b/hi");
    await call(env, "/", {
      method: "POST",
      body: JSON.stringify({ input: ["a", "b"], labels: ["x", "y"] }),
      key: "key-a",
    });
    expect((await call(env, "/a,b/hi")).status).toBe(429);
    await call(env, "/a,b/hi", { ip: "198.51.100.9" });
    const calls = cloudflareSql();
    const html = await (await call(env, "/stats")).text();
    expect(html).toContain(
      "<h2>Today (UTC), live</h2><div class=\"tiles\"><div><b>4</b>calls</div><div><b>5</b>texts classified</div>" +
        "<div><b>2</b>users</div><div><b>1</b>with an unlimited key</div><div><b>1</b>hit the daily limit</div>" +
        "<div><b>0</b>failed</div></div>",
    );
    expect(html).toMatch(/<h2>Last 7 days, updated \d\d:\d\d UTC<\/h2>/);
    // The next call shows at once; the tables aren't queried again.
    expect(calls).toHaveLength(8);
    runpod(DONE);
    await call(env, "/a,b/hi", { ip: "192.0.2.4" });
    const again = cloudflareSql();
    expect(await (await call(env, "/stats")).text()).toContain("<b>5</b>calls");
    expect(again).toHaveLength(0);
  });

  it("saves each call as one small write, keeps it through a restart, and starts over each day", async () => {
    const store = storage();
    const state = { storage: store } as unknown as DurableObjectState;
    const live = new LiveStats(state, {});
    for (let i = 0; i < 10; i++) {
      await live.add({ units: 2, status: i ? 200 : 502, keyed: false, client: String(i % 3).repeat(32) });
    }
    await live.add({ units: 1, status: 200, keyed: true, client: "none" });
    expect(store.writes).toBe(11);
    expect([...store.data.keys()]).toEqual(["today"]);
    // A new instance, as after the object sat idle or a deploy, has everything.
    const restarted = new LiveStats(state, {});
    expect((await restarted.page(7)).today).toMatchObject({ calls: 11, texts: 21, users: 3, keyed: 1, failed: 1 });
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 86_400_000);
    expect((await restarted.page(7)).today).toMatchObject({ calls: 0, users: 0 });
  });

  it("keeps nothing past its UTC day", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const at = (time: string) => vi.setSystemTime(new Date(time));
    const one = { units: 1, status: 200, keyed: false, client: "a".repeat(32) };
    at("2026-09-21T23:00:00Z");
    const store = storage();
    const live = new LiveStats({ storage: store } as unknown as DurableObjectState, {});
    await live.add(one);
    await live.page(90);
    expect([...store.data.keys()].sort()).toEqual(["tables:90", "today"]);
    expect(await store.getAlarm()).toBe(Date.parse("2026-09-22T00:00:00Z"));
    at("2026-09-22T00:00:00Z");
    await store.deleteAlarm();
    await live.alarm();
    expect(store.data.size).toBe(0);
    expect(await store.getAlarm()).toBeNull();

    // A call just after midnight, before the alarm runs, has started the new day: that stays,
    // and goes the midnight after.
    at("2026-09-22T23:59:59Z");
    await live.add(one);
    at("2026-09-23T00:00:00.500Z");
    await live.add(one);
    await store.deleteAlarm();
    await live.alarm();
    expect(store.data.get("today")).toMatchObject({ day: "2026-09-23", calls: 1 });
    expect(await store.getAlarm()).toBe(Date.parse("2026-09-24T00:00:00Z"));
  });

  it("counts users exactly while they are few, then estimates them in a fixed size", async () => {
    const store = storage();
    const live = new LiveStats({ storage: store } as unknown as DurableObjectState, {});
    // Stand-ins for client IDs: fixed, and as uniform as the real HMACs.
    const ids = await Promise.all(
      Array.from({ length: 2000 }, async (_, n) => {
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(n)));
        return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
      }),
    );
    const add = (client: string) => live.add({ units: 1, status: 200, keyed: false, client });
    for (let i = 0; i < 1500; i++) await add(ids[i % 1000]);
    expect((await live.page(7)).today.users).toBe(1000);
    for (let i = 1000; i < 2000; i++) await add(ids[i]);
    expect(Math.abs((await live.page(7)).today.users - 2000)).toBeLessThan(60);
    // Past 1,024 the IDs are dropped for 4,096 small ranks, however many more come.
    const saved = store.data.get("today") as { ids: number[]; sketch: Uint8Array };
    expect(saved.ids).toEqual([]);
    expect(saved.sketch.length).toBe(4096);
    expect(Math.max(...saved.sketch)).toBeLessThanOrEqual(21);
  });

  it("queries the chosen range and shows the numbers, escaped", async () => {
    const calls = cloudflareSql();
    const html = await (await call(makeEnv(connected), "/stats?days=30")).text();
    const [first] = calls;
    expect(first.url).toBe("https://api.cloudflare.com/client/v4/accounts/acc1/analytics_engine/sql");
    expect(new Headers(first.init?.headers).get("Authorization")).toBe("Bearer cf-read");
    for (const { init } of calls) {
      expect(String(init?.body)).toContain("INTERVAL '30' DAY");
      expect(String(init?.body)).toMatch(/FORMAT JSON$/);
      // Claimed calling websites are recorded but never shown.
      expect(String(init?.body)).not.toContain("blob7");
    }
    expect(html).toContain("<b>1,234</b>calls");
    expect(html).toContain("<b>1,303</b>ms median response");
    expect(html).toContain("<td>2026-09-21</td>");
    expect(html).toContain("Calls per day (UTC)");
    expect(html).toContain("Calls per hour (UTC)");
    expect(html).toContain('role="img"');
    expect(html).toContain("data-tip=");
    expect(html).toContain('id="chart-tip"');
    expect(html).toContain("<b>30 days</b>");
    expect(html).toContain("&#60;script&#62;");
    expect(html).not.toContain("<script>x");
    expect(html).not.toContain("cf-read");
  });

  it("caches the page 30 s per data center, and the tables ten minutes for all of them", async () => {
    const store = workersCache();
    const calls = cloudflareSql();
    const env = makeEnv(connected);
    const first = await call(env, "/stats");
    expect(first.headers.get("cache-control")).toBe("public, max-age=30");
    expect(calls).toHaveLength(8);
    await call(env, "/stats?days=7");
    await call(env, "/stats?days=5;DROP");
    expect(calls).toHaveLength(8);
    await call(env, "/stats?days=90");
    expect(calls).toHaveLength(16);
    expect([...store.keys()]).toEqual([
      "https://sifty.example/stats?days=7",
      "https://sifty.example/stats?days=90",
    ]);
    // Another data center, or this one 30 s on, renders the page again but shares the tables.
    store.clear();
    await call(env, "/stats");
    expect(calls).toHaveLength(16);
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 600_000);
    store.clear();
    await call(env, "/stats");
    expect(calls).toHaveLength(24);
  });

  it("queries a range once when views arrive together", async () => {
    const calls = cloudflareSql();
    const env = makeEnv(connected);
    await Promise.all([call(env, "/stats"), call(env, "/stats"), call(env, "/stats")]);
    expect(calls).toHaveLength(8);
  });

  it("says so when not connected or failing, without the details, and retries after a minute", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const calls = cloudflareSql(false);
    const unset = await call(makeEnv(), "/stats");
    expect(await unset.text()).toContain("aren't connected yet");
    expect(calls).toHaveLength(0);
    const env = makeEnv(connected);
    const failing = await call(env, "/stats");
    expect(failing.status).toBe(200);
    const text = await failing.text();
    expect(text).toContain("unavailable right now");
    expect(text).toContain("Today (UTC), live");
    expect(text).not.toContain("acc1");
    expect(calls).toHaveLength(8);
    await call(env, "/stats");
    expect(calls).toHaveLength(8);
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 60_000);
    await call(env, "/stats");
    expect(calls).toHaveLength(16);
    // Even with the live object down, the page answers.
    const down = await call(makeEnv({ LIVE: refusingLive() }), "/stats");
    expect(down.status).toBe(200);
    expect(await down.text()).toContain("unavailable right now");
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
