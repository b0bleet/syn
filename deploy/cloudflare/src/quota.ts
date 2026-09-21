import { DurableObject } from "cloudflare:workers";

interface Usage {
  day: string;
  used: number;
}

/**
 * A daily usage counter. One instance per client (named by a hash of its IP) plus one named
 * "global" for the whole free tier. Calls to one instance run one at a time, so a read, check,
 * and write cannot interleave with another request's. Storage is wiped at the end of the day.
 */
export class Quota extends DurableObject {
  /** Charge `units` for `day` if they fit under `limit`. */
  async take(
    units: number,
    limit: number,
    day: string,
    resetAt: number,
  ): Promise<{ allowed: boolean; remaining: number }> {
    const used = await this.used(day);
    if (used + units > limit) return { allowed: false, remaining: Math.max(0, limit - used) };
    await this.ctx.storage.put<Usage>("usage", { day, used: used + units });
    if ((await this.ctx.storage.getAlarm()) === null) await this.ctx.storage.setAlarm(resetAt);
    return { allowed: true, remaining: limit - used - units };
  }

  /** Give back units charged for work that then failed on our side. */
  async refund(units: number, day: string): Promise<void> {
    const used = await this.used(day);
    if (used > 0) await this.ctx.storage.put<Usage>("usage", { day, used: Math.max(0, used - units) });
  }

  /** The day is over: keep nothing about this client. */
  async alarm(): Promise<void> {
    await this.ctx.storage.deleteAll();
  }

  private async used(day: string): Promise<number> {
    const usage = await this.ctx.storage.get<Usage>("usage");
    return usage?.day === day ? usage.used : 0;
  }
}
