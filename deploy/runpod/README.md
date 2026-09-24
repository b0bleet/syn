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
- **GPU**: 24 GB is enough for `Qwen/Qwen3.5-4B` in bfloat16. 48 GB for 14B, 80 GB for 32B.
  Keep the serverless endpoint on RTX 4090 / A40 / RTX A6000. On 2026-09-24 its workers on
  the 16 GB and 24 GB Ampere pools (RTX A4000, A4500, RTX 4000 Ada, RTX A5000) restarted
  every few seconds and never took a job, with any image, on this endpoint and on a fresh one;
  the same image took jobs at once on the larger cards. The always-on pod (see
  `scripts/runpod_serve.py`) is not affected and runs fine on a community RTX A4500.
- **Model** (Manage → Edit Endpoint): set it to the same value as `SYN_MODEL`, e.g.
  `Qwen/Qwen3.5-4B`. RunPod then caches the weights on the host at
  `/runpod-volume/huggingface-cache`, which is the image's `HF_HOME`, so workers load
  from disk instead of downloading ~16 GB on every cold start. Free, and download time
  is not billed. A network volume works too but is slower to read.
- **Env vars**: `SYN_MODEL` (default `Qwen/Qwen3.5-4B`), `SYN_READOUT` (`letters`),
  `SYN_ORDERINGS` (`1`, the caller's option order only), `SYN_REVISION` to pin a commit.
  The SGLang-side settings are unused here. The pmi, head, and pointer readouts stay on Qwen3.

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

## Train the general head on a pod

`scripts/runpod_train.py` starts a GPU pod that clones this repository, caches features for
every dataset under `data/`, runs `syn transfer`, and stops itself. Nothing runs on your
machine, and nothing needs a volume: a private Hugging Face repo is the store.

```sh
export RUNPOD_API_KEY=...            # console -> Settings -> API Keys
export HF_TOKEN=...                  # huggingface.co/settings/tokens, write access
uv run python scripts/hf_store.py push data --repo <user>/syn-training   # optional: your own rows
uv run python scripts/runpod_train.py --model Qwen/Qwen3-8B --hf-repo <user>/syn-training --wait
```

- **The repo** (`--hf-repo`, created private if missing) holds `data/<source>/*.jsonl` (your
  rows and imported suites; the public datasets are rebuilt on the pod), `features/<model>/`
  (cached once per backbone, pushed as soon as they are extracted, pulled by the next run),
  `heads/<model>/<run>/` (checkpoints, `transfer.json`, `RESULT.md`), and `logs/`. Serve a
  head from it with `SYN_READOUT=head` and
  `SYN_HEAD_PATH=hf://<user>/syn-training/heads/<model>/<run>/all-sources.safetensors`, plus
  `HF_TOKEN` on the endpoint; the worker downloads the two small files at startup.
  `scripts/hf_store.py pull heads/<model>/<run> --repo <user>/syn-training --to runs` fetches
  a run to your machine.
- **A network volume** (`--volume-id`) still works instead of, or as well as, the repo; then
  the same files sit under `/runpod-volume` and `SYN_HEAD_PATH` can be a path on it. Without
  either, results stay on the pod's disk under `/workspace` until it is stopped.
- `--suites DIR,...` imports extra System One labelled-request suites (see
  `syn import-systemone`) before caching features; `--sources` restricts the datasets.
- `--ref` picks the commit the pod runs; it defaults to your current HEAD, so push first.
- The `RESULT.md` table has one row per source: in-domain accuracy from the all-sources head,
  transfer accuracy from the head trained without that source, and the shuffled-context
  control. The transfer column is the one that says whether the head is general.
- Memory: cached features are about a megabyte per row at 8B, held in RAM while the heads
  train; `--limit-per-source` (default 2000) bounds that. `--gpu` defaults to an L40S.
- The pod runs `deploy/runpod/train_job.py`, which also works on any GPU box with the
  `[local,hub]` extras. On failure the pod stays up so its logs can be read; stop it with
  `--stop <pod-id>`.
- `--task pointer` adapts the backbone and trains the pointer head. It reads
  `pointer-data/<source>/{train,validation,calibration}.jsonl` from the store, scores
  `pointer-data/transfer-dev.jsonl` and each source's `test.jsonl`, and pushes the merged
  backbone plus `pointer.safetensors` to `pointers/<model>/<run>/`. `--epochs` applies
  here too (default 2). Adapter rank 16, batch 2 with 4 accumulation steps, and gradient
  checkpointing. The anchor is off unless `--anchor` is passed: its teacher is the letters
  readout in the chat prompt, while the pointer learns a cloze prefix. Pointer training now
  defaults to cross-entropy only (`--ordinal-weight 0`), ordinary row weights, and checkpoint
  selection on in-domain validation. The transfer file is only scored at the end of each run;
  repeated experimentation makes it a development set, not an untouched final test.
  `--balance-sources`, `--holdout-selection`, and `--policy-cases` are independent opt-in
  experiments. Policy copies only append date counts; the automatic uniform-label generator
  was removed because it could delete irrelevant evidence and teach an incorrect target.
  `--ordinal-weight 1` restores the anchor-off run's ordinal loss. The head task's ordinal
  default remains 1. Use `--limit-per-source 0` to match the complete 14,576-row pointer run.
  The checkpoint sidecar records experiment flags, training-row counts and input-file hashes.
  Serve from the store with
  `SYN_MODEL=hf://<user>/<repo>/pointers/<model>/<run>/backbone`,
  `SYN_READOUT=pointer`, and
  `SYN_POINTER_PATH=hf://<user>/<repo>/pointers/<model>/<run>/pointer.safetensors`.
  The head task does not read `pointer-data/`.
- `pointer-data/README.md` in the store records where those rows came from. Choice questions
  with more than 26 options are dropped on import. Structured state and criteria are flattened
  to labeled lines, the same rendering `syn serve` uses.

## Next controlled pointer experiment

Wait for the current anchor-off result before allocating another pod. After committing and
pushing the changes, preview the CE-only experiment (the preview makes no API request):

```sh
uv run python scripts/runpod_train.py --task pointer \
  --model Qwen/Qwen3-8B-Base --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --hf-repo jolobuild/syn-training --limit-per-source 0 --epochs 2 \
  --ordinal-weight 0 --run qwen3-8b-ce-only --dry-run
```

This uses rank 16, learning rate `5e-5`, batch 2, accumulation 4, seed 7, maximum 1,024
tokens, and the existing none/distractor augmentation. Anchor, balancing, source holdout,
and date copies are off. Compare with the anchor-off run before changing another setting.
The pinned base revision matches the earlier 8B checkpoint; verify the anchor-off checkpoint
has that revision and the same input data before calling this a controlled comparison.

For a paired serving comparison, serve a pinned reference Qwen3-8B checkpoint on port 8009 and
the trained SYN pointer on port 8765, then use the same converted rows for both:

```sh
uv run syn bench data/pointer-upload/transfer-dev.jsonl --out runs/reference-paired \
  --target reference@reference-latest=http://127.0.0.1:8009 \
  --target syn=http://127.0.0.1:8765
```

The report includes overall and per-source accuracy, calibration metrics, paired bootstrap
intervals, row failures and the dataset hash. Use a new output directory for each checkpoint
pair: resume verifies rows and endpoint/model names, not the weights behind a running server.
Record checkpoint revisions separately. Require all 764 rows to score before claiming a win.
This protocol sends each converted row as one question, with binary questions represented
as choice. It is a common comparison protocol, not a reproduction of the reference model's published native
suite scores. Evaluate an untouched transfer test only after all model selection is complete.

## Notes

- GitHub import: set the Dockerfile path to `deploy/runpod/Dockerfile` and the build
  context to the repo root. RunPod's scanner only looks for `runpod.serverless.start()`
  at the repo root, so it warns about `deploy/runpod/handler.py`; continue anyway.
- Cold start = image pull + model load + weight download. The public API avoids it with an
  always-on pod (`scripts/runpod_serve.py`, see `deploy/cloudflare/README.md`) and keeps this
  endpoint at zero active workers as the fallback.
- `readout=head` needs `SYN_HEAD_PATH` pointing at a checkpoint inside the image or the
  network volume, and the head must have been trained on the same backbone; the local
  heads were trained on 0.6B features and will not load against an 8B model.
- The letters readout is what the measured numbers belong to: at 8B it scored 0.825 on
  sms_spam and 0.868 on AG News zero-shot, where 0.6B scored 0.495 and 0.740.
