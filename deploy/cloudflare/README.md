# Cloudflare Worker

The public front door: a web page, and a free API with daily limits. It holds no API logic.
It meters the request, sends it to the RunPod endpoint as the job input
`{"http": {method, path, headers, body}}`, polls through GPU cold starts, and returns the
status, headers, and body the Python app produced on the GPU. Every route (`/<labels>/<text>`,
`/?labels=&text=`, `POST /`, `/v1/score`, `/v1/systemone`, `/v1/models`) therefore behaves
exactly as `syn serve` does locally.

- **Page**: `/` is `public/index.html`, with a playground that calls the API on the same
  origin, for browsers and link-preview bots alike; only command-line tools such as curl that
  don't ask for HTML get the app's plain-text usage. The page's other files (preview image,
  icons, `robots.txt`, `sitemap.xml`, `llms.txt`) are served as they are, uncounted. They are
  listed in `STATIC_FILES` in `src/index.ts`; keep that list in step with `public/`, since any
  other path runs a GPU job. `robots.txt` keeps crawlers to the page and these files.
  `scripts/page_images.py` redraws the icons and preview image. CORS is open, so other sites can
  call the API too.
- **Free tier**: requests without a key are counted per client IP (IPv6 by /64, stored only as
  a hash) per UTC day, and against a global daily cap that bounds GPU spend. One unit per
  request; a batch counts each text, a System One call each question. Responses carry
  `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`; over the limit is a
  `429` with `Retry-After`. Units are refunded when the failure is on our side. Counters live
  in the `Quota` Durable Object and delete themselves at the end of the day.
- **Keys**: bearer keys in `API_KEYS` skip both limits. Any other key is treated as the free
  tier, so clients that require a key (typesafe-sdk) work with `api_key="free"`.
- `/health` reports RunPod worker counts, uncounted and without waking a GPU.
- **Statistics**: every API call writes one data point to the `syn_api_calls` Analytics Engine
  dataset: endpoint, status, free or keyed, source (`playground`, another `website` and its
  host, or `direct`), client kind (curl, python, typesafe-sdk, browser, ...), country, texts
  classified, latency, and RunPod's queue and GPU time. Never the text, the labels, or the IP
  address. Distinct clients are counted per day by an ID keyed with a Worker secret and the
  date, so it can't be reversed or linked across days. Kept three months; the free plan allows
  100,000 points a day. The page and its files aren't counted.

## Statistics

`/stats` is a public page, linked from the site. It shows the last 24 hours, 7, 30, or 90
days: totals, then per day (calls, texts, distinct users, keyed calls, `429`s, failures, median
and p95 latency), endpoints, sources and clients, countries, status codes, and GPU time with cold
starts. Calling websites are recorded but not shown, since any script can claim any `Origin`.
Each range is cached for ten minutes, which keeps it within the free plan's 10,000 queries a day
however many people look. It reads the dataset through Cloudflare's SQL API, so the Worker needs
two more secrets; the token stays in the Worker:

```sh
npx wrangler secret put STATS_ACCOUNT_ID   # Workers & Pages overview, right-hand column
npx wrangler secret put STATS_API_TOKEN    # a token with only "Account Analytics: Read"
```

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
