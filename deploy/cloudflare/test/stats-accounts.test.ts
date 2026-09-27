import { afterAll, beforeAll, expect, it, vi } from "vitest";
import { Miniflare, convertV4MiniflareOptions } from "miniflare";
import { readFileSync } from "node:fs";
import { registrationStats, statsPage } from "../src/stats";
import type { LiveStats } from "../src/live";

let mf: Miniflare;
let db: D1Database;
beforeAll(async () => {
  mf = new Miniflare(convertV4MiniflareOptions({ workers: [{ name: "registration-stats-test", modules: true, script: "export default {fetch(){return new Response('ok')}}", compatibilityDate: "2026-09-01", d1Databases: ["AUTH_DB"] }] }));
  db = await mf.getD1Database("AUTH_DB") as unknown as D1Database;
  const sql = readFileSync(new URL("../migrations/0001_accounts.sql", import.meta.url).pathname, "utf8");
  await db.exec(sql.replace(/--[^\n]*/g, "").replace(/\n/g, " "));
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

it("keeps account statistics available when API statistics fail and vice versa", async () => {
  const log = vi.spyOn(console, "error").mockImplementation(() => {});
  try {
    const failedLive = {page: async () => {throw new Error("private failure");}} as unknown as DurableObjectStub<LiveStats>;
    const html = await (await statsPage(new URL("https://example.com/stats"), failedLive, db)).text();
    expect(html).toContain("<b>3</b>accounts in total");
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
