# Cloudflare Worker

The public front door: a web page, and a free API with daily limits. It holds no API logic.
It meters the request, sends it to the RunPod endpoint as the job input
`{"http": {method, path, headers, body}}`, polls through GPU cold starts, and returns the
status, headers, and body the Python app produced on the GPU. Every route (`/<labels>/<text>`,
`/?labels=&text=`, `POST /`, `/v1/score`, `/v1/systemone`, `/v1/models`) therefore behaves
exactly as `syn serve` does locally.

- **Page**: browsers asking for `/` get `public/index.html`, with a playground that calls the
  API on the same origin. CORS is open, so other sites can call the API too.
- **Free tier**: requests without a key are counted per client IP (IPv6 by /64, stored only as
  a hash) per UTC day, and against a global daily cap that bounds GPU spend. One unit per
  request; a batch counts each text, a System One call each question. Responses carry
  `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`; over the limit is a
  `429` with `Retry-After`. Units are refunded when the failure is on our side. Counters live
  in the `Quota` Durable Object and delete themselves at the end of the day.
- **Keys**: bearer keys in `API_KEYS` skip both limits. Any other key is treated as the free
  tier, so clients that require a key (typesafe-sdk) work with `api_key="free"`.
- `/health` reports RunPod worker counts, uncounted and without waking a GPU.

## Deploy

```sh
cd deploy/cloudflare
npm ci
npx wrangler login                         # once, opens a browser
npx wrangler secret put RUNPOD_ENDPOINT_ID # your RunPod endpoint id
npx wrangler secret put RUNPOD_API_KEY     # your RunPod key
npx wrangler secret put API_KEYS           # optional: comma-separated keys with no limits
npx wrangler deploy                        # -> https://syn.<your-subdomain>.workers.dev
```

Secrets persist across deploys, so this is once per account; CI (`.github/workflows/ci.yml`)
redeploys the code on later pushes. Limits are `DAILY_LIMIT` and `GLOBAL_DAILY_LIMIT` in
`wrangler.jsonc` (the page states `DAILY_LIMIT`; keep them in step). For your own domain, add a
custom domain to the Worker in the Cloudflare dashboard (the zone must be on Cloudflare). Leave
`SYN_API_KEY` unset on the RunPod endpoint: the endpoint only accepts the RunPod key.

## Limits of the platform

A Worker has no wall-clock limit while the caller stays connected, so cold starts are fine.
The free plan allows 50 outgoing requests per incoming one; `JOB_TIMEOUT_SECONDS=240` with
`POLL_SECONDS=5` stays under that. A job still pending at the deadline is cancelled and the
caller gets a 504.

## Test

```sh
npm test          # vitest, RunPod and the Durable Object stubbed
npm run check     # tsc
npx wrangler dev  # local; RUNPOD_API_BASE in .dev.vars points it at a stand-in
```
