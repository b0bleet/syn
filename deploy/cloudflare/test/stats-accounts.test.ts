import { afterAll, beforeAll, expect, it, vi } from "vitest";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
import { readdirSync, readFileSync } from "node:fs";
import { registrationStats, statsPage } from "../src/stats";
import type { LiveStats } from "../src/live";

let mf: Miniflare;
let db: D1Database;
beforeAll(async () => {
  mf = new Miniflare(convertV4MiniflareOptions({ workers: [{ name: "registration-stats-test", modules: true, script: "export default {fetch(){return new Response('ok')}}", compatibilityDate: "2026-09-01", d1Databases: ["AUTH_DB"] }] }));
  db = await mf.getD1Database("AUTH_DB") as unknown as D1Database;
  const dir = new URL("../migrations/", import.meta.url).pathname;
  for (const file of readdirSync(dir).sort()) {
    await db.exec(readFileSync(dir + file, "utf8").replace(/--[^\n]*/g, "").replace(/\n/g, " "));
  }
});
afterAll(async () => { await mf?.dispose(); });

it("renders zero registrations without inventing users", async () => {
  const html = await registrationStats(db, 7);
  expect(html).toContain("<b>0</b>accounts in total");
  expect(html).toContain("<b>0</b>email verified");
  expect(html).toContain("None in this period.");
});

it("counts accounts and non-expiring keys, and groups recent registrations by UTC date", async () => {
  const now = Date.now(), old = now - 40 * 86400_000;
  for (const [id, verified, at] of [["old", 1, old], ["new", 1, now], ["pending", 0, now]] as const) {
    await db.prepare("INSERT INTO user(id,name,email,emailVerified,createdAt,updatedAt) VALUES (?,?,?,?,?,?)")
      .bind(id, "Private Name", `${id}@private.example`, verified, at, at).run();
  }
  await db.prepare("INSERT INTO api_key(userId,digest,suffix,expiresAt) VALUES (?,?,?,?)").bind("old", "private-digest", "secret", 1).run();
  const html = await registrationStats(db, 7);
  expect(html).toContain("<b>3</b>accounts in total");
  expect(html).toContain("<b>2</b>email verified");
  expect(html).toContain("<b>1</b>awaiting verification");
  expect(html).toContain("<b>1</b>with an active API key");
  expect(html).toContain("<b>2</b>registered today (UTC)");
  expect(html).toContain("<b>2</b>registered in last 7 days");
  expect(html).toContain(new Date(now).toISOString().slice(0,10));
  expect(html).not.toContain(new Date(old).toISOString().slice(0,10));
  expect(await registrationStats(db, 90)).toContain("<b>3</b>registered in last 90 days");
  for (const value of ["Private Name", "private.example", "private-digest", "secret"]) expect(html).not.toContain(value);
});

it("follows signups from source and method to first answer and API key", async () => {
  const now = Date.now();
  await db.batch([
    ["hn-1", "hacker news", 1, now, "github"], ["hn-2", "hacker news", 1, null, "credential"],
    ["hn-3", "hacker news", 0, null, "credential"], ["rd-1", "reddit", 1, now, "credential"],
  ].map(([id, source, verified, called, provider]) => [
    db.prepare("INSERT INTO user(id,name,email,emailVerified,createdAt,updatedAt,signupSource,firstCallAt) VALUES (?,?,?,?,?,?,?,?)")
      .bind(id, "Private Name", `${id}@private.example`, verified, now, now, source, called),
    db.prepare("INSERT INTO account(id,accountId,providerId,userId,createdAt,updatedAt) VALUES (?,?,?,?,?,?)")
      .bind(`a-${id}`, `a-${id}`, provider, id, now, now),
  ]).flat());
  await db.prepare("INSERT INTO api_key(userId,digest,suffix,expiresAt) VALUES (?,?,?,?)").bind("hn-1", "digest-hn", "abc123", 0).run();
  const html = await registrationStats(db, 7);
  // In the last 7 days: new, pending (no source) and the four above.
  expect(html).toContain("<div><b>6</b>registered</div><div><b>4</b>verified</div><div><b>2</b>got a first answer</div><div><b>1</b>have an API key</div>");
  expect(html).toContain("<h2>Where signups came from</h2>");
  expect(html).toMatch(/<td>hacker news<\/td><td class="bar"><span[^>]*><\/span>3<\/td><td>2<\/td><td>1<\/td><td>1<\/td>/);
  expect(html).toMatch(/<td>not recorded<\/td><td class="bar"><span[^>]*><\/span>2<\/td><td>1<\/td><td>0<\/td><td>0<\/td>/);
  expect(html).toMatch(/<td>reddit<\/td><td class="bar"><span[^>]*><\/span>1<\/td><td>1<\/td><td>1<\/td><td>0<\/td>/);
  expect(html).toMatch(/<td>github<\/td><td class="bar"><span[^>]*><\/span>1<\/td>/);
  expect(html).toMatch(/<td>email<\/td><td class="bar"><span[^>]*><\/span>5<\/td>/);
  for (const value of ["Private Name", "private.example", "digest-hn", "abc123"]) expect(html).not.toContain(value);
});

it("keeps account statistics available when API statistics fail and vice versa", async () => {
  const log = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const failedLive = {page: async () => {throw new Error("private failure");}} as unknown as DurableObjectStub<LiveStats>;
    const html = await (await statsPage(new URL("https://example.com/stats"), failedLive, db)).text();
    expect(html).toContain("<b>7</b>accounts in total");
    expect(html).toContain("Statistics are unavailable");
    expect(html).not.toContain("private failure");
    const failedDb = {batch: async () => {throw new Error("private DB failure");}, prepare: () => ({bind: () => ({})})} as unknown as D1Database;
    const live = {page: async () => ({today: {calls: 9, texts: 9, users: 1, keyed: 0, limited: 0, failed: 0}, tables: {at: Date.now(), ok: false, body: ""}})} as unknown as DurableObjectStub<LiveStats>;
    const partial = await (await statsPage(new URL("https://example.com/stats"), live, failedDb)).text();
    expect(partial).toContain("Registration statistics are unavailable");
    expect(partial).toContain("<b>9</b>calls");
    expect(partial).not.toContain("private DB failure");
  } finally { log.mockRestore(); }
});
