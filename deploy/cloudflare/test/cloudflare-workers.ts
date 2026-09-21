/** Test stand-in for the `cloudflare:workers` module: just the Durable Object base class. */
export class DurableObject<Env = unknown> {
  constructor(
    protected ctx: DurableObjectState,
    protected env: Env,
  ) {}
}
