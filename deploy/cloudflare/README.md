# Cloudflare Worker

The public front door: a web page, and a free API with daily limits. It holds no API logic.
It meters the request and sends it to the always-on GPU pod (`POD_URL`, see below). When the
pod is unset, down, or failing, it sends the request to the RunPod serverless endpoint instead,
as the job input `{"http": {method, path, headers, body}}`, and polls through GPU cold starts.
Either way it returns the status, headers, and body the Python app produced on the GPU. Every
route (`/<labels>/<text>`,
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
- **Always-on pod**: an on-demand community-cloud pod runs `syn serve` behind `SYN_API_KEY`
  (`scripts/runpod_serve.py` creates it). It costs about a quarter of an always-on serverless
  worker and answers in well under a second, so the serverless endpoint can scale to zero and
  only takes calls when the pod can't: unreachable, slower than `POD_TIMEOUT_SECONDS` (20),
  a server error, a refused key, or a non-JSON error from RunPod's proxy while the pod is
  stopped or starting. Secrets: `POD_URL` (`https://<pod id>-8765.proxy.runpod.net`) and
  `POD_API_KEY`. Without `POD_URL`, every call goes to the serverless endpoint.
- `/health` reports whether the pod is up and the RunPod worker counts, uncounted and without
  waking a GPU. It stays `ready` while the pod alone can answer.
- **Statistics**: every API call writes one data point to the `syn_api_calls` Analytics Engine
  dataset: endpoint, status, free or keyed, source (`playground`, another `website` and its
  host, or `direct`), client kind (curl, python, typesafe-sdk, browser, ...), country, texts
  classified, latency, and RunPod's queue and GPU time. Never the text, the labels, or the IP
  address. Distinct clients are counted per day by an ID keyed with a Worker secret and the
  date, so it can't be reversed or linked across days. Kept three months; the free plan allows
  100,000 points a day. The same call is added to today's live counts. Both happen after the
  answer is sent, so they never slow it. The page and its files aren't counted.

## Private request dashboard (free)

`REQUEST_LOGS=true` writes a structured `sifty.request` event when a classification payload is
received, before quota checks or GPU work, and a second event when the API returns. Both share
a `request_id`; `phase` is `received` or `completed`. A receipt without a completion can mean
the request is still pending or the caller disconnected. Cloudflare can buffer/index logs after
the invocation ends: this is not an instant queue monitor. Open
**Cloudflare dashboard → Workers & Pages → syn → Observability**, select
**Last 24 hours**, and filter the `event` field to `sifty.request`. Expand a record to see its
`request` and `response`; filter by `country`, `client`, `source`, `endpoint`, or `status`.
The Message column identifies the phase and endpoint. Use `phase = "completed"` for one row
per finished call, or `phase = "received"` for submitted payloads. Use **Actions → Refresh**
to fetch new stored events; **Live** follows new invocations. The table time is the log emission
time, while the JSON `timestamp` is when the request arrived (UTC).
Access is controlled by your Cloudflare account permissions; there is no public read endpoint.
Public `/stats` continues to use content-free Analytics Engine events and aggregate live counts.

Records include submitted text, labels/questions, the result, timestamp, country, region,
city, timezone, network ASN/provider, Cloudflare data center, client family, source website
hostname (when supplied), tier, units, status, and request/queue/GPU durations. Location comes
from the network connection and may reflect a VPN or server. Headers, credentials, raw IPs,
full referrer URLs, and unrelated query/body fields are not copied into the structured event.
Sensitive information included in classification text itself is still content: do not submit it.
Only classification routes create these events: GET `/<labels>/<text>`, GET `/?text=...&labels=...`,
POST `/` (including every input in a batch), POST `/v1/score`, and POST `/v1/systemone`.
Health, static pages, contact submissions, `/stats`, model discovery, documentation, unrelated
routes, and HEAD/OPTIONS requests do not create request records. Quota refusals and backend
failures on classification routes record their payload and status.

Normal payloads are JSON objects. Each request/response longer than 12,000 serialized characters
becomes a preview marked `truncated`; bodies over 128,000 characters are omitted with a reason.
Malformed JSON is marked `invalid_json`. These limits affect logs only, never the API payload.

Workers Free includes **200,000 log events per account per day, retained for 3 days**. This setup
adds no paid subscription, KV, D1, or log export service. Invocation logs are disabled to avoid
duplicate events and automatic raw-URL capture; errors still produce logs. The free allowance
is shared across the account, and collection can stop at platform limits. The 20,000-unit API
cap does not cover refused requests, keyed traffic, or other Workers. A **Last 24 hours** filter
does not delete the other two days. Workers Paid has 7-day retention and different billing;
keep Workers Free to avoid overage charges and update the public notice if that ever changes.
See [Workers Logs pricing](https://developers.cloudflare.com/workers/observability/logs/workers-logs/#pricing).

Set `REQUEST_LOGS=false` and deploy to stop content collection. Existing logs expire according
to Cloudflare's retention. Only requests after deployment will have these structured records.

## Statistics

`/stats` is a public page, linked from the site. At the top are today's counts (UTC), live:
calls, texts, distinct users, keyed calls, `429`s, and failures. Below are tables for the last 24
hours, 7, 30, or 90 days: totals, then per day (the same counts plus median and p95 latency),
endpoints, sources and clients, countries, status codes, and GPU time with cold starts. Endpoints
are the app's own routes; any other path is named by its shape, and calling websites are recorded
but not shown, so no caller can put words on the page.

Both come from the `LiveStats` Durable Object, one for every data center. Today's counts are one
stored value, saved on every call (an idle object loses its memory, so they can't wait there),
which costs one row write a call. On the free plan all Durable Objects share 100,000 row writes a
day and `Quota` must never run out; with its two or so per free call, the 20,000-unit daily cap
comes to about 60,000. Users are counted exactly up to 1,024 a day from 32 bits of each client
ID, then estimated within about 2% by a 4 KB sketch, so the value stays small. Counting starts
from the first deploy that has it. The tables are queried at most every ten minutes per range
however many data centers ask, about 4,000 of the free plan's 10,000 queries a day. The page is
cached for 30 seconds per data center, and an open page refreshes itself every 30 seconds for
half an hour, since every refresh is a Worker request.

Old statistics expire on their own. Cloudflare deletes Analytics Engine data after three months,
which can't be shortened or deleted early; the page reads 90 days at most.
`LiveStats` and `Quota` clear themselves at midnight UTC, and Worker logs last three days.

The tables read the dataset through Cloudflare's SQL API, so the Worker needs one more secret;
the account id lives in `wrangler.jsonc` and the token stays in the Worker:

```sh
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
npm test          # vitest, RunPod and the Durable Objects stubbed
npm run check     # tsc
npx wrangler dev  # local; RUNPOD_API_BASE in .dev.vars points it at a stand-in
```
