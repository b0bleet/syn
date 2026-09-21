# Cloudflare Worker

The public API. It holds no API logic: it checks the caller's bearer key, sends the request to
the RunPod endpoint as the job input `{"http": {method, path, headers, body}}`, polls through
GPU cold starts, and returns the status, headers, and body the Python app produced on the GPU.
Every route (`/<labels>/<text>`, `/?labels=&text=`, `POST /`, `/v1/score`, `/v1/systemone`,
`/v1/models`) therefore behaves exactly as `syn serve` does locally. `/health` reports RunPod
worker counts without a key and without waking a GPU.

## Deploy

```sh
cd deploy/cloudflare
npm ci
npx wrangler login                         # once, opens a browser
npx wrangler secret put RUNPOD_ENDPOINT_ID # your RunPod endpoint id
npx wrangler secret put RUNPOD_API_KEY     # your RunPod key
npx wrangler secret put API_KEYS           # comma-separated keys your callers send
npx wrangler deploy                        # -> https://syn.<your-subdomain>.workers.dev
```

Secrets persist across deploys, so this is once per account; CI (`.github/workflows/ci.yml`)
redeploys the code on later pushes. For your own domain, add a route or custom domain in
`wrangler.jsonc` (the zone must be on Cloudflare). Leave `SYN_API_KEY` unset on the RunPod
endpoint: callers are authenticated here, and the endpoint only accepts the RunPod key.

```sh
curl -H "Authorization: Bearer <key>" "https://syn.<your-subdomain>.workers.dev/spam,ham/Win+a+free+iPhone"
```

## Limits

A Worker has no wall-clock limit while the caller stays connected, so cold starts are fine.
The free plan allows 50 outgoing requests per incoming one; `JOB_TIMEOUT_SECONDS=240` with
`POLL_SECONDS=5` stays under that. A job still pending at the deadline is cancelled and the
caller gets a 504.

## Test

```sh
npm test          # vitest, RunPod stubbed
npm run check     # tsc
npx wrangler dev  # local; RUNPOD_API_BASE in .dev.vars points it at a stand-in
```
