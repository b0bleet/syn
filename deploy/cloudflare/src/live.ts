import { DurableObject } from "cloudflare:workers";
import { queryTables, RANGES, type StatsEnv, type Tables } from "./stats";

/** One API call, as today's counts see it. */
export interface Call {
  units: number;
  status: number;
  keyed: boolean;
  /** The call's daily client ID, or "none" without the secret that keys it (local dev). */
  client: string;
}

/** Today's totals, per UTC day. */
export interface Today {
  day: string;
  calls: number;
  texts: number;
  users: number;
  keyed: number;
  limited: number;
  failed: number;
}

interface Saved extends Omit<Today, "users"> {
  /** The first 32 bits of each of today's client IDs, while there are at most EXACT of them. */
  ids: number[];
  /** Past that, a HyperLogLog sketch of them instead. */
  sketch: Uint8Array | null;
}

// Users are counted exactly up to this many a day, then estimated.
const EXACT = 1024;
// Sketch registers, picked by an ID's top 12 bits; about 1.6% error.
const REGISTERS = 4096;
// A range's tables are queried again after this long, or sooner after a failure.
const TABLES_MS = 600_000;
const RETRY_MS = 60_000;

/**
 * What /stats shows, in one place for every data center: today's counts, live, and the Analytics
 * Engine tables for each range.
 *
 * Every call is saved as it comes: an idle object loses its memory within seconds, so counts
 * can't wait there to be saved in batches. All of today is one stored value, so a call costs one
 * row write. That matters because on the free plan all Durable Objects share 100,000 row writes a
 * day, and Quota failing would fail the API: Quota's two or so per free call plus this one keep
 * the 20,000-unit daily cap near 60,000. Users are counted exactly from 32 bits of each daily
 * client ID while there are at most 1,024, then with a 4 KB sketch, so the value stays small
 * however many come. Like the IDs themselves, neither can be linked across days, and like Quota,
 * nothing here outlives its UTC day: an alarm clears it at midnight.
 *
 * The Workers cache is per data center, so tables cached there would be queried once per data
 * center. Here each range is queried at most every ten minutes wherever the page is viewed:
 * 4 ranges x 144 x 7 queries, about 4,000 of the 10,000 a day the free plan allows.
 */
export class LiveStats extends DurableObject<StatsEnv> {
  private held = new Map<number, Tables>();
  private refreshing = new Map<number, Promise<Tables>>();

  /** Count one API call. */
  async add(call: Call): Promise<void> {
    const today = await this.today();
    today.calls += 1;
    today.texts += call.units;
    if (call.keyed) today.keyed += 1;
    if (call.status === 429) today.limited += 1;
    if (call.status >= 500) today.failed += 1;
    if (call.client !== "none") see(today, Number.parseInt(call.client.slice(0, 8), 16));
    await this.ctx.storage.put<Saved>("today", today);
    await this.clearAtMidnight();
  }

  /** Everything the page shows for a range of `days`. */
  async page(days: number): Promise<{ today: Today; tables: Tables }> {
    const [{ ids, sketch, ...today }, tables] = await Promise.all([this.today(), this.tables(days)]);
    return { today: { ...today, users: sketch ? estimate(sketch) : ids.length }, tables };
  }

  /**
   * Midnight UTC: the day's counts go, with the user IDs in them, and so do the stored tables,
   * which are queried again when next viewed. A call just after midnight may already have
   * started the new day; that is kept, and cleared the next midnight.
   */
  async alarm(): Promise<void> {
    const saved = await this.ctx.storage.get<Saved>("today");
    const kept = saved?.day === utcDay();
    if (saved && !kept) await this.ctx.storage.delete("today");
    await this.ctx.storage.delete(RANGES.map((days) => `tables:${days}`));
    this.held.clear();
    if (kept) await this.ctx.storage.setAlarm(nextMidnight());
  }

  private async clearAtMidnight(): Promise<void> {
    if ((await this.ctx.storage.getAlarm()) === null) await this.ctx.storage.setAlarm(nextMidnight());
  }

  /** Today's counts, started afresh each UTC day. */
  private async today(): Promise<Saved> {
    const day = utcDay();
    const saved = await this.ctx.storage.get<Saved>("today");
    if (saved?.day === day) return saved;
    return { day, calls: 0, texts: 0, keyed: 0, limited: 0, failed: 0, ids: [], sketch: null };
  }

  /** A range's tables, queried again once stale, and once however many ask at the same time. */
  private async tables(days: number): Promise<Tables> {
    let held = this.held.get(days);
    if (!held) {
      held = await this.ctx.storage.get<Tables>(`tables:${days}`);
      if (held) this.held.set(days, held);
    }
    if (held && Date.now() - held.at < (held.ok ? TABLES_MS : RETRY_MS)) return held;
    let pending = this.refreshing.get(days);
    if (!pending) {
      pending = this.refresh(days).finally(() => this.refreshing.delete(days));
      this.refreshing.set(days, pending);
    }
    return pending;
  }

  private async refresh(days: number): Promise<Tables> {
    const tables = await queryTables(days, this.env);
    this.held.set(days, tables);
    await this.ctx.storage.put(`tables:${days}`, tables);
    await this.clearAtMidnight();
    return tables;
  }
}

function utcDay(): string {
  return new Date().toISOString().slice(0, 10);
}

function nextMidnight(): number {
  const now = new Date();
  return Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
}

/** Count a user by 32 bits of their client ID: in the list while it is short, then the sketch. */
function see(today: Saved, id: number): void {
  if (!today.sketch) {
    if (today.ids.includes(id)) return;
    today.ids.push(id);
    if (today.ids.length <= EXACT) return;
    today.sketch = new Uint8Array(REGISTERS);
    for (const known of today.ids) mark(today.sketch, known);
    today.ids = [];
    return;
  }
  mark(today.sketch, id);
}

/**
 * Mark an ID in a HyperLogLog sketch. The ID comes from an HMAC, so its bits are uniform: the top
 * 12 pick a register, and the rank is where the first 1 falls in the other 20 (21 if none does).
 */
function mark(registers: Uint8Array, id: number): void {
  const rank = Math.clz32((id << 12) | 0x800) + 1;
  if (rank > registers[id >>> 20]) registers[id >>> 20] = rank;
}

/** Distinct IDs marked in the sketch: within about 2%, and near exact while they are few. */
function estimate(registers: Uint8Array): number {
  const m = registers.length;
  let sum = 0;
  let empty = 0;
  for (const rank of registers) {
    sum += 2 ** -rank;
    if (rank === 0) empty += 1;
  }
  const raw = ((0.7213 / (1 + 1.079 / m)) * m * m) / sum;
  // While most registers are empty, counting the empty ones is more accurate.
  return Math.round(raw <= 2.5 * m && empty > 0 ? m * Math.log(m / empty) : raw);
}
