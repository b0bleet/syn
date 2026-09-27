import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
import { readFileSync } from "node:fs";
import { authenticated, authRoute, type AuthEnv } from "../src/auth";
import worker, { type Env } from "../src/index";

const base = "https://sifty.example";
let mf: Miniflare;
let env: AuthEnv;
let db: D1Database;
const mail: { to: string[]; text: string }[] = [];
const pending: Promise<unknown>[] = [];
const ctx = { waitUntil: (p: Promise<unknown>) => pending.push(p) } as unknown as ExecutionContext;
const password = "correct horse battery staple!";

beforeAll(async () => {
  mf = new Miniflare(convertV4MiniflareOptions({ workers: [{ name: "auth-test", modules: true, script: "export default {fetch(){return new Response('ok')}}", compatibilityDate: "2026-09-01", d1Databases: ["AUTH_DB"] }] }));
  db = await mf.getD1Database("AUTH_DB") as unknown as D1Database;
  for (const file of ["0001_accounts.sql", "0002_signup_measurement.sql", "0003_signup_conversion_id.sql"]) {
    const sql = readFileSync(new URL("../migrations/"+file, import.meta.url).pathname, "utf8");
    await db.exec(sql.replace(/--[^\n]*/g, "").replace(/\n/g, " "));
  }
  env = {
    AUTH_DB: db, BETTER_AUTH_SECRET: "test-only-secret-at-least-thirty-two-characters", AUTH_BASE_URL: base,
    RESEND_API_KEY: "test-resend",
    QUOTA: { idFromName: (n: string) => n, get: () => ({ take: async () => ({allowed:true}) }) } as unknown as AuthEnv["QUOTA"],
  };
});
afterAll(async () => { await mf?.dispose(); });
beforeEach(async () => { await db.prepare('DELETE FROM rateLimit').run(); });
afterEach(() => { vi.unstubAllGlobals(); });

function mockMail() {
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
    expect(url).toBe("https://api.resend.com/emails");
    mail.push(JSON.parse(String(init.body)));
    return Response.json({id:"test-email"});
  }));
}
async function route(path: string, body?: object, cookie?: string, method?: string, extraHeaders: Record<string,string> = {}) {
  const request = new Request(new URL(path, base), {
    method: method ?? (body ? "POST" : "GET"),
    headers: { Origin: base, "Content-Type": "application/json", "CF-Connecting-IP": "192.0.2.1", ...(cookie ? {Cookie:cookie}: {}), ...extraHeaders },
    ...(body ? {body: JSON.stringify(body)} : {}),
  });
  const response = await authRoute(request, env, ctx);
  await Promise.all(pending.splice(0));
  return response;
}
function cookies(response: Response) {
  return response.headers.getSetCookie().map(value=>value.split(";")[0]).join("; ");
}
function mailURL() { return mail.at(-1)!.text.match(/https:\/\/\S+/)![0]; }
async function signedIn(email: string, fromAd = false) {
  mockMail();
  expect((await route("/api/auth/sign-up/email", {name:"Test",email,password}, undefined, undefined, fromAd ? {"X-Sifty-Reddit-Visit":"1"} : {})).status).toBe(200);
  expect((await route(mailURL())).status).toBe(302);
  const login = await route("/api/auth/sign-in/email", {email,password});
  expect(login.status).toBe(200);
  return cookies(login);
}

describe("verified registration and hosted API gate", () => {
  it("requires verification, rejects forged access, and rotates/revokes hashed keys", async () => {
    mockMail();
    const email="verify@example.com";
    const signup=await route("/api/auth/sign-up/email",{name:"Test",email,password});
    expect(signup.status).toBe(200);
    expect(signup.headers.get("set-cookie")).toBeNull();
    expect(mail.at(-1)!.to).toEqual([email]);
    const url=mailURL();
    expect((await route("/api/auth/sign-in/email",{email,password})).status).toBe(403);
    expect((await route("/api/account/key",{})).status).toBe(401);
    expect(await authenticated(new Request(base+"/",{method:"POST",headers:{Origin:base,"Content-Type":"application/json",Cookie:"better-auth.session_token=forged"}}),env)).toBe(false);
    expect((await route(url)).status).toBe(302);
    expect((await route("/api/auth/sign-in/email",{email,password:"incorrect password"})).status).toBe(401);
    const login=await route("/api/auth/sign-in/email",{email,password});
    expect(login.status).toBe(200);
    expect(login.headers.get("set-cookie")).toContain("HttpOnly");
    expect(login.headers.get("set-cookie")).toContain("Secure");
    const cookie=cookies(login);
    const req=(origin=base)=>new Request(base+"/",{method:"POST",headers:{Origin:origin,"Content-Type":"application/json",Cookie:cookie}});
    expect(await authenticated(req(),env)).toBe(true);
    expect(await authenticated(req("https://evil.example"),env)).toBe(false);
    const key1=await (await route("/api/account/key",{},cookie)).json() as {key:string};
    const withKey=(key:string)=>new Request(base+"/a,b/test",{headers:{Authorization:`Bearer ${key}`}});
    expect(await authenticated(withKey(key1.key),env)).toBe(true);
    expect(key1).not.toHaveProperty("expiresAt");
    // Previously issued keys remain usable even after their old expiration date.
    await db.prepare("UPDATE api_key SET expiresAt = 1").run();
    expect(await authenticated(withKey(key1.key),env)).toBe(true);
    expect(await (await route("/api/account",undefined,cookie)).json()).toMatchObject({key:{suffix:key1.key.slice(-6)}});
    expect((await (await route("/api/account",undefined,cookie)).json() as {key:object}).key).not.toHaveProperty("expiresAt");
    const stored=await db.prepare("SELECT digest FROM api_key").first<{digest:string}>();
    expect(stored!.digest).not.toBe(key1.key);
    const key2=await (await route("/api/account/key",{},cookie)).json() as {key:string};
    expect(await authenticated(withKey(key1.key),env)).toBe(false);
    expect(await authenticated(withKey(key2.key),env)).toBe(true);
    expect((await route("/api/account/key",{},cookie,"DELETE")).status).toBe(200);
    expect(await authenticated(withKey(key2.key),env)).toBe(false);
    await route("/api/auth/sign-out",{},cookie);
    expect(await authenticated(req(),env)).toBe(false);
  });

  it("resets passwords once and revokes sessions and API keys",async()=>{
    const email="reset@example.com", cookie=await signedIn(email);
    const key=await (await route("/api/account/key",{},cookie)).json() as {key:string};
    await route("/api/auth/request-password-reset",{email,redirectTo:base+"/account"});
    const redirect=await route(mailURL());
    const token=new URL(redirect.headers.get("location")!).searchParams.get("token");
    expect(token).toBeTruthy();
    expect((await route("/api/auth/reset-password",{token,newPassword:"replacement secure password"})).status).toBe(200);
    expect((await route("/api/auth/reset-password",{token,newPassword:password})).status).not.toBe(200);
    expect((await route("/api/account",undefined,cookie)).status).toBe(401);
    expect(await authenticated(new Request(base+"/",{headers:{Authorization:`Bearer ${key.key}`}}),env)).toBe(false);
    expect((await route("/api/auth/sign-in/email",{email,password})).status).toBe(401);
    expect((await route("/api/auth/sign-in/email",{email,password:"replacement secure password"})).status).toBe(200);
  });

  it("signs the user in from the verification link and returns to the callback",async()=>{
    mockMail();
    const email="welcome@example.com";
    await route("/api/auth/sign-up/email",{name:"Test",email,password,callbackURL:base+"/account?verified=1"});
    const verified=await route(mailURL());
    expect(verified.status).toBe(302);
    expect(verified.headers.get("location")).toBe(base+"/account?verified=1");
    const cookie=cookies(verified);
    expect(await (await route("/api/account",undefined,cookie)).json()).toMatchObject({email,key:null});
    expect(await authenticated(new Request(base+"/",{method:"POST",headers:{Origin:base,"Content-Type":"application/json",Cookie:cookie}}),env)).toBe(true);
    // A used link cannot sign anyone in again.
    expect(cookies(await route(mailURL()))).toBe("");
  });

  it("refuses invalid verification and enforces password length",async()=>{
    mockMail();
    expect((await route("/api/auth/sign-up/email",{name:"Test",email:"short@example.com",password:"short"})).status).toBe(400);
    expect((await route("/api/auth/sign-up/email",{name:"Test",email:"eight@example.com",password:"eight888"})).status).toBe(200);
    expect((await route("/api/auth/verify-email?token=invalid")).status).not.toBe(200);
  });

  it("limits repeated sign-in attempts",async()=>{
    mockMail();
    for (let i=0;i<5;i++) expect((await route("/api/auth/sign-in/email",{email:"missing@example.com",password})).status).toBe(401);
    expect((await route("/api/auth/sign-in/email",{email:"missing@example.com",password})).status).toBe(429);
  });

  it("blocks cross-origin requests, exposes only configured providers, and fails closed",async()=>{
    const response=await authRoute(new Request(base+"/api/auth/sign-up/email",{method:"POST",headers:{Origin:"https://evil.example"},body:"{}"}),env,ctx);
    expect(response.status).toBe(403);
    expect(await (await route("/api/auth/providers")).json()).toEqual({github:false});
    expect(await authenticated(new Request(base+"/"),{QUOTA:env.QUOTA})).toBe(false);
    expect((await authRoute(new Request(base+"/api/auth/get-session"),{QUOTA:env.QUOTA},ctx)).status).toBe(503);
  });

  it("never invokes the GPU or logs inputs for anonymous or invalid-key requests",async()=>{
    const fetch=vi.fn(); vi.stubGlobal("fetch",fetch);
    const log=vi.spyOn(console,"log");
    for(const path of ["/","/v1/systemone","/spam,ham/hello","/?labels=a,b&text=hello"]) {
      const response=await worker.fetch(new Request(base+path,{method:"POST",headers:{Authorization:"Bearer free"},body:'{"input":"private text"}'}),{...env,AUTH_REQUIRED:"true",REQUEST_LOGS:"true"} as Env,ctx);
      expect(response.status).toBe(401);
    }
    expect(fetch).not.toHaveBeenCalled(); expect(log).not.toHaveBeenCalled(); log.mockRestore();
  });
});

// Count only real new accounts acquired from an ad, after email verification and sign-in.
describe("SignUp conversion claims", () => {
  it("claims a verified ad signup once, including across repeat sessions", async () => {
    const cookie = await signedIn("ad-signup@example.com", true);
    const account = await (await route("/api/account", undefined, cookie)).json() as {signupConversionEligible:boolean};
    expect(account.signupConversionEligible).toBe(true);
    const claim = await (await route("/api/account/signup-conversion", {}, cookie)).json() as {report:boolean; conversionId:string};
    expect(claim).toEqual({report:true, conversionId:expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)});
    expect(await (await route("/api/account/signup-conversion", {}, cookie)).json()).toEqual({report:false});
    expect(await db.prepare("SELECT redditSignupConversionId FROM user WHERE email = ?").bind("ad-signup@example.com").first()).toEqual({redditSignupConversionId:claim.conversionId});
    const anotherCookie = await signedIn("another-ad-signup@example.com", true);
    const anotherClaim = await (await route("/api/account/signup-conversion", {}, anotherCookie)).json() as {conversionId:string};
    expect(anotherClaim.conversionId).not.toBe(claim.conversionId);
    expect(await (await route("/api/account", undefined, cookie)).json()).toMatchObject({signupConversionEligible:false});
  });
  it("excludes direct signups and prevents duplicate signup attempts from reattributing them", async () => {
    const email = "direct-signup@example.com", cookie = await signedIn(email);
    await route("/api/auth/sign-up/email", {name:"Test",email,password}, undefined, undefined, {"X-Sifty-Reddit-Visit":"1"});
    expect(await (await route("/api/account/signup-conversion", {}, cookie)).json()).toEqual({report:false});
    expect((await route("/api/account/signup-conversion", {})).status).toBe(401);
  });
  it("honors privacy headers and expires old signup eligibility", async () => {
    const cookie = await signedIn("private-signup@example.com", true);
    expect(await (await route("/api/account/signup-conversion", {}, cookie, undefined, {"Sec-GPC":"1"})).json()).toEqual({report:false});
    await db.prepare("UPDATE user SET createdAt = 0 WHERE email = ?").bind("private-signup@example.com").run();
    expect(await (await route("/api/account/signup-conversion", {}, cookie)).json()).toEqual({report:false});
  });
});
