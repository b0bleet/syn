# RunPod serverless deploy

One GPU worker loads `SYN_MODEL` once and scores one request per job, the same `Scorer`
the local service uses, so measured behaviour carries over. No SGLang needed; the local
backend runs in-process on the worker's GPU.

## Build and push

```sh
docker build -f deploy/runpod/Dockerfile -t <dockerhub-user>/syn-scorer .
docker push <dockerhub-user>/syn-scorer
```

## Endpoint

RunPod console → Serverless → New Endpoint:

- **Image**: `<dockerhub-user>/syn-scorer`
- **GPU**: 24 GB is enough for `Qwen/Qwen3-8B` in bfloat16. 48 GB for 14B, 80 GB for 32B.
- **Model** (Manage → Edit Endpoint): set it to the same value as `SYN_MODEL`, e.g.
  `Qwen/Qwen3-8B`. RunPod then caches the weights on the host at
  `/runpod-volume/huggingface-cache`, which is the image's `HF_HOME`, so workers load
  from disk instead of downloading ~16 GB on every cold start. Free, and download time
  is not billed. A network volume works too but is slower to read.
- **Env vars**: `SYN_MODEL` (default `Qwen/Qwen3-8B`), `SYN_READOUT` (`letters`),
  `SYN_REVISION` to pin a commit. The SGLang-side settings are unused here.

## Call it

`POST https://api.runpod.ai/v2/<endpoint-id>/runsync` with `Authorization: Bearer
<RUNPOD_API_KEY>`:

```json
{"input": {"context": "Win a free iPhone",
           "question": "Which label best applies to the text?",
           "options": [{"id": "spam", "text": "spam"}, {"id": "not spam", "text": "not spam"}]}}
```

or the URL shorthand:

```json
{"input": {"url": "/spam,not+spam/Win+a+free+iPhone"}}
```

`output.selected_option_id` is the answer; the full `ScoreResponse` (probabilities, wins,
agreement, latency) comes back verbatim. A bad payload fails the job with the reason in `error`.

The public API is the Cloudflare Worker in `deploy/cloudflare/`, which sends
`{"input": {"http": {"method", "path", "headers", "body"}}}`; the handler replays that request
against the same FastAPI app `syn serve` runs and returns `{"status", "headers", "body"}`.

## Notes

- GitHub import: set the Dockerfile path to `deploy/runpod/Dockerfile` and the build
  context to the repo root. RunPod's scanner only looks for `runpod.serverless.start()`
  at the repo root, so it warns about `deploy/runpod/handler.py`; continue anyway.
- Cold start = image pull + model load + weight download. Keep at least one worker warm,
  or accept a few minutes on the first job.
- `readout=head` needs `SYN_HEAD_PATH` pointing at a checkpoint inside the image or the
  network volume, and the head must have been trained on the same backbone; the local
  heads were trained on 0.6B features and will not load against an 8B model.
- The letters readout is what the measured numbers belong to: at 8B it scored 0.825 on
  sms_spam and 0.868 on AG News zero-shot, where 0.6B scored 0.495 and 0.740.
