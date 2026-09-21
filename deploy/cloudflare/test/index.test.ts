import { afterEach, describe, expect, it, vi } from "vitest";
import worker, { type Env } from "../src/index";

const env: Env = {
  RUNPOD_ENDPOINT_ID: "ep1",
  RUNPOD_API_KEY: "rp-secret",
  API_KEYS: "key-a, key-b",
  JOB_TIMEOUT_SECONDS: "60",
  POLL_SECONDS: "0",
};
const RUNPOD = "https://api.runpod.ai/v2/ep1";
const OK = { status: 200, headers: { "content-type": "text/plain" }, body: "spam\n" };

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

function call(path: string, init: RequestInit = {}, key: string | null = "key-b") {
  const headers = new Headers(init.headers);
  if (key) headers.set("Authorization", `Bearer ${key}`);
  return worker.fetch(new Request(`https://syn.example${path}`, { ...init, headers }), env);
}

afterEach(() => vi.unstubAllGlobals());

describe("auth", () => {
  it("rejects a missing, malformed, or unknown key before calling RunPod", async () => {
    const calls = runpod();
    for (const key of [null, "nope"]) expect((await call("/a,b/hi", {}, key)).status).toBe(401);
    const basic = await worker.fetch(
      new Request("https://syn.example/a,b/hi", { headers: { Authorization: "Basic key-a" } }),
      env,
    );
    expect(basic.status).toBe(401);
    expect(basic.headers.get("WWW-Authenticate")).toBe("Bearer");
    expect(calls).toHaveLength(0);
  });

  it("accepts any configured key", async () => {
    runpod({ id: "j", status: "COMPLETED", output: OK }, { id: "j", status: "COMPLETED", output: OK });
    expect((await call("/a,b/hi", {}, "key-a")).status).toBe(200);
    expect((await call("/a,b/hi", {}, "key-b")).status).toBe(200);
  });
});

describe("proxy", () => {
  it("wraps the raw request as a RunPod job and returns the app's response", async () => {
    const calls = runpod({
      id: "j",
      status: "COMPLETED",
      output: { status: 200, headers: { "content-type": "text/plain", "x-syn-selected": "spam" }, body: "spam\n" },
    });
    const response = await call("/spam,a%2Cb/Win+a+free+iPhone?verbose=1");
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
      output: { status: 422, headers: { "content-type": "application/json" }, body: '{"detail":"bad"}' },
    });
    const body = JSON.stringify({ state: "hi", model: "m", questions: {} });
    const response = await call("/v1/systemone", {
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
      { id: "j9", status: "COMPLETED", output: OK },
    );
    expect(await (await call("/a,b/hi")).text()).toBe("spam\n");
    expect(calls.map((c) => c.url)).toEqual([
      `${RUNPOD}/runsync`,
      `${RUNPOD}/status/j9`,
      `${RUNPOD}/status/j9`,
    ]);
  });

  it("cancels the job and answers 504 at the deadline", async () => {
    const calls = runpod({ id: "j9", status: "IN_QUEUE" }, {});
    const response = await worker.fetch(
      new Request("https://syn.example/a,b/hi", { headers: { Authorization: "Bearer key-a" } }),
      { ...env, JOB_TIMEOUT_SECONDS: "0" },
    );
    expect(response.status).toBe(504);
    expect(calls.at(-1)).toMatchObject({ url: `${RUNPOD}/cancel/j9` });
  });

  it("answers 502 without leaking worker errors", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    runpod({ id: "j", status: "FAILED", error: "Traceback: /secret/path.py" });
    const failed = await call("/a,b/hi");
    expect(failed.status).toBe(502);
    expect(await failed.text()).not.toContain("secret");

    runpod(new Response("nope", { status: 401 }));
    expect(await (await call("/a,b/hi")).json()).toEqual({ detail: "RunPod answered 401" });

    runpod({ id: "j", status: "COMPLETED", output: { unexpected: true } });
    expect((await call("/a,b/hi")).status).toBe(502);
  });
});

describe("health", () => {
  it("reports RunPod worker counts without a key or a job", async () => {
    const calls = runpod({ workers: { idle: 1, running: 0 }, jobs: { inQueue: 0 } });
    const response = await call("/health", {}, null);
    expect(await response.json()).toEqual({
      status: "ready",
      workers: { idle: 1, running: 0 },
      jobs: { inQueue: 0 },
    });
    expect(calls.map((c) => c.url)).toEqual([`${RUNPOD}/health`]);
  });

  it("is degraded when RunPod does not answer", async () => {
    runpod(new Response("down", { status: 500 }));
    expect((await call("/health", {}, null)).status).toBe(503);
  });
});
