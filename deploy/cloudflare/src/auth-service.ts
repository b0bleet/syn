import { DurableObject } from "cloudflare:workers";
import { authRoute, type AuthEnv } from "./auth";

/** Password hashing needs more CPU than a Free-plan edge request's 10 ms allowance. */
export class AuthService extends DurableObject<AuthEnv> {
  // Bound concurrent scrypt memory use. D1 remains the persistent account store.
  private pending: Promise<unknown> = Promise.resolve();

  async fetch(request: Request): Promise<Response> {
    const task = this.pending.then(() => authRoute(request, this.env, this.ctx));
    this.pending = task.catch(() => {});
    return task;
  }
}
