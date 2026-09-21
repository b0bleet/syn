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

async function call(env: Env, path: string, init: RequestInit & { ip?: string; key?: string } = {}) {
  const headers = new Headers(init.headers);
  headers.set("CF-Connecting-IP", init.ip ?? "203.0.113.7");
  if (init.key) headers.set("Authorization", `Bearer ${init.key}`);
  // What the Worker finishes after answering, such as recording the call, is done before a test looks.
  const pending: Promise<unknown>[] = [];
  const ctx = { waitUntil: (promise: Promise<unknown>) => void pending.push(promise) };
  const request = new Request(`https://sifty.example${path}`, { ...init, headers });
  const response = await worker.fetch(request, env, ctx as unknown as ExecutionContext);
  await Promise.all(pending);
  return response;
}

async function detail(response: Response): Promise<string> {
  return ((await response.json()) as { detail: string }).detail;
}

afterEach(() => {
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
    for (const path of STATIC_FILES) {
      const response = await call(env, path);
      expect(await response.text()).toBe("file");
      expect(response.headers.get("X-RateLimit-Remaining")).toBeNull();
    }
    expect(served).toEqual([...STATIC_FILES]);
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
    expect(calls.length).toBe(7);
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
    expect(calls).toHaveLength(7);
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
    expect(calls).toHaveLength(7);
    await call(env, "/stats?days=7");
    await call(env, "/stats?days=5;DROP");
    expect(calls).toHaveLength(7);
    await call(env, "/stats?days=90");
    expect(calls).toHaveLength(14);
    expect([...store.keys()]).toEqual([
      "https://sifty.example/stats?days=7",
      "https://sifty.example/stats?days=90",
    ]);
    // Another data center, or this one 30 s on, renders the page again but shares the tables.
    store.clear();
    await call(env, "/stats");
    expect(calls).toHaveLength(14);
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 600_000);
    store.clear();
    await call(env, "/stats");
    expect(calls).toHaveLength(21);
  });

  it("queries a range once when views arrive together", async () => {
    const calls = cloudflareSql();
    const env = makeEnv(connected);
    await Promise.all([call(env, "/stats"), call(env, "/stats"), call(env, "/stats")]);
    expect(calls).toHaveLength(7);
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
    expect(calls).toHaveLength(7);
    await call(env, "/stats");
    expect(calls).toHaveLength(7);
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 60_000);
    await call(env, "/stats");
    expect(calls).toHaveLength(14);
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
