# syn

Zero-shot option scoring on pretrained Qwen3. No answer generation, no training required. Send a context, a question, and 2-26 options; get back a probability per option, the selected id, and abstention signals.

## Run

Requires Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra local --locked
HF_HOME=.cache/huggingface uv run --extra local syn serve   # 127.0.0.1:8765
```

First run downloads Qwen3-0.6B (~1.5 GB). To prefetch weights or pull a larger checkpoint:

```sh
uv run python scripts/download_models.py                # $SYN_MODEL or Qwen/Qwen3-0.6B
uv run python scripts/download_models.py Qwen/Qwen3-8B  # supports model@revision
```

```sh
curl -sS http://127.0.0.1:8765/v1/score -H 'Content-Type: application/json' -d '{
  "context": "My card was charged twice and I want the duplicate refunded",
  "question": "Which team should handle this?",
  "options": [
    {"id": "billing", "text": "Billing and refunds"},
    {"id": "technical", "text": "Technical support"},
    {"id": "sales", "text": "Sales enquiries"}
  ]
}'
```

URL shortcuts. GET answers with the bare label; `?verbose=1` returns `{model, results: [{label, confidence, scores, ms}], usage}`, `?format=json` the full response above:

```sh
curl "http://127.0.0.1:8765/spam,ham/Win+a+free+iPhone"                  # -> spam
curl "http://127.0.0.1:8765/?labels=spam,ham&text=Win+a+free+iPhone&verbose=1"
curl "http://127.0.0.1:8765/billing,technical,sales/I+was+charged+twice?q=Which+team%3F"
curl http://127.0.0.1:8765/ -H 'Content-Type: application/json' \
  -d '{"input": ["Win a free iPhone", "Lunch at noon?"], "labels": ["spam", "ham"]}'   # batch of up to 32
```

`+` means a space; in the path form percent-encode `%2C`, `%2B`, `%2F` inside labels. Interactive API docs at `/docs`, health at `/health`. A GET puts the classified text in the URL, which lands in access logs, so don't send personal data through it.

## How it scores

- **letters** (default): builds N cyclic option orderings so every option occupies every position once, reads label log-probs, averages, softmaxes. `ordering_agreement` exposes ordering disagreement, the signal that catches confidently wrong answers.
- **pmi**: scores each option's text likelihood given the context, prior-corrected, in one masked forward. Order-invariant by construction. Helps with short label words; weak on long descriptive options.
- **head**: a trained cross-attention head over frozen backbone features (`syn train-head`); local backend only, needs `SYN_HEAD_PATH`.
- **pointer**: a backbone adapted by `syn train-pointer` (low-rank adapters, merged into the weights) read by a trained pointer head over delimited, isolated option spans. Order-invariant by construction; local backend only, needs `SYN_POINTER_PATH`.

The service abstains (`selected_option_id: null`, `abstain_reasons`) below the `SYN_ABSTAIN_THRESHOLD`, `SYN_MIN_CONFIDENCE`, and `SYN_MIN_ORDERING_AGREEMENT` gates. Probabilities are option preferences, not calibrated success rates.

Label wording changes answers more than anything else. Use neutral, parallel labels and verify any label set on your own examples. `SYN_PROMPT_FORMAT=text` renders labelled sections instead of JSON for multi-line states.

## Data

`scripts/download_data.py` rebuilds all of `data/`: the HF imports used for the measurements below, the synthetic routing set, and the `data/eval/` subsets:

```sh
uv run --extra hub python scripts/download_data.py
uv run syn import-hf fancyzhx/ag_news --out data/agnews --all-options   # custom imports
uv run syn synthetic --out data/synthetic                              # routing set generator
```

## Evaluate and train

```sh
uv run syn evaluate data/synthetic/test.jsonl \
  --output runs/synthetic.jsonl --shuffle-options > runs/metrics.json
uv run syn compare runs/a.jsonl runs/b.jsonl     # paired bootstrap diff
```

Trained head (frozen features cached once; the ~1M-param head trains in seconds, on the GPU when there is one):

```sh
uv run --extra local syn features data/agnews/train.jsonl --out runs/feats/agnews-train.npz  # + validation, calibration, test
uv run --extra local syn train-head runs/feats/agnews-train.npz runs/feats/banking77-train.npz \
  --validation runs/feats/agnews-validation.npz runs/feats/banking77-validation.npz \
  --out runs/head.safetensors --p-none 0.1 --p-none-distract 0.12 --p-distract 0.15
uv run --extra local syn eval-head runs/head.safetensors runs/feats/agnews-test.npz
SYN_READOUT=head SYN_HEAD_PATH=runs/head.safetensors uv run --extra local syn serve
```

Several feature files train one head. Every row carries a source tag (`--source`, default the dataset's directory name), so `eval-head` reports per source as well. The temperature is fitted on the `--calibration` rows (or validation) and stored in the checkpoint; `serve` applies it unless `SYN_TEMPERATURE` is set. `--p-none`, `--p-none-distract`, and `--p-distract` add "None of the above" wordings and unrelated distractors to training rows, so the head learns when nothing offered fits and that an extra irrelevant option changes nothing. Rows marked `"ordinal": true` (score questions, options are ordered levels) get a ranked-probability loss on top (`--ordinal-weight`).

`eval-head` reports a shuffled-context control alongside real accuracy; if the control doesn't collapse toward chance, the head is reading option priors, not the state. Whether a head is general is the transfer number, not in-domain accuracy:

```sh
uv run --extra local syn transfer --train runs/feats/*-train.npz --validation runs/feats/*-validation.npz \
  --test runs/feats/*-test.npz --out runs/transfer
```

trains one head on all sources and one without each source, then tests every held-out head on the source it never saw; `transfer.json` lists in-domain, transfer, and control accuracy per source. `scripts/runpod_train.py` runs the whole pipeline on a RunPod GPU (see `deploy/runpod/README.md`).

System One suites, labelled `/v1/systemone` requests (a `state`, `questions` with a `label` each), import as rows with `syn import-systemone <dir> --out data/<name>`; score questions come out ordinal, and rows render exactly as the service renders them.

Adapted backbone (GPU, `--extra train`): `syn train-pointer` adds low-rank adapters to the backbone and trains a pointer head over delimited, isolated option spans, with the same augmentation, the ordinal loss, and an optional anchor to the base model's zero-shot distribution (`--anchor`, a `syn evaluate` output on the training rows). The adapters are merged into the weights on save, so serving needs only the `local` extra:

```sh
uv run --extra local --extra train syn train-pointer data/agnews/train.jsonl data/banking77/train.jsonl \
  --validation data/agnews/validation.jsonl data/banking77/validation.jsonl --out runs/pointer
SYN_MODEL=runs/pointer/backbone SYN_READOUT=pointer SYN_POINTER_PATH=runs/pointer/pointer.safetensors \
  uv run --extra local syn serve
```

## Benchmark against Jev

`syn bench` sends every labeled row to each target as the same System One request, one choice question with the row's options, and scores all targets the same way: accuracy with a 95% interval, log loss, Brier, ECE, latency, and a paired bootstrap of each target against the first. Any server that speaks `POST /v1/systemone` can be a target, this one included.

```sh
export TYPESAFE_API_KEY=...   # from console.typesafe.ai/keys
uv run syn bench data/eval/agnews-250.jsonl data/eval/banking77-250.jsonl data/spam-eval.jsonl \
  --out runs/bench --target jev --target syn=http://127.0.0.1:8765
```

A target is `NAME[@MODEL][=URL]`. `jev` means `https://api.typesafe.ai` with `jev-latest`; pin a version with `jev@<version>`, since the version that answered is recorded per row. Other targets send `<NAME>_API_KEY` as a bearer token when it's set. Rows marked `"ordinal": true` go as score questions with their options as the levels, and `level_mae` reports how far each target's expected level sat from the true one. Rows land in `runs/bench/<dataset>/<target>.jsonl`, with `summary.json` and `summary.md` beside them. Re-running resumes: scored rows are kept, failed rows are retried, and a target added later only scores its own rows.

Requests go one at a time by default, so latency is one round trip from your machine, network included. `--concurrency N` is faster but adds queueing at the target to the latency. Public datasets may be in any model's training data, so confirm a result on your own labeled rows before relying on it.

## SGLang backend

```sh
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
SYN_BACKEND=sglang SYN_MODEL=Qwen/Qwen3-8B uv run syn serve
```

Startup refuses a remote running a different model or revision; `/health` probes it and returns 503 `degraded` when unreachable.

## TypeSafe SDK

`POST /v1/systemone` and `GET /v1/models` speak the [typesafe-sdk](https://pypi.org/project/typesafe-sdk/) protocol, so its clients work unchanged. Each question is scored as options: `Noul` is yes versus no (`noul` is P(yes)), `Choice` one option per criteria key, `Score` one option per rubric level (`score` is the probability-weighted level). Any `model` name is accepted.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="...", base_url="http://127.0.0.1:8765", model="syn-latest", timeout=120)
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this ticket?", criteria=["can wait", "this week", "today"]),
    },
)
response.nouls["billing"].noul, response.choices["tone"].choice, response.scores["urgency"].score
```

The SDK's default timeout is 10 s; pass `timeout=120` when the GPU may be cold.

## Public API

A Cloudflare Worker (`deploy/cloudflare/`) is the public front door: a web page with a playground, and a free API. It counts each client IP against a daily allowance (1,000 units by default) and a global daily cap, forwards the request as a RunPod job, polls through GPU cold starts, and returns what the same FastAPI app produced on the GPU (`deploy/runpod/`, scales to zero). The first request after scaling to zero waits for a worker (about a minute); warm requests take about half a second.

CI (`.github/workflows/ci.yml`) tests every push and pull request. On `main` it deploys what changed: code under `src/` or `deploy/runpod/` becomes a GitHub release, which RunPod rebuilds from, and `deploy/cloudflare/` is redeployed with wrangler. Deploys stay off until the repository sets the variables `DEPLOY_GPU` / `DEPLOY_WORKER` to `true` (and the `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID` secrets for the Worker), so forks never deploy.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `SYN_BACKEND` | `local` | `local` or `sglang` |
| `SYN_MODEL` | `Qwen/Qwen3-0.6B` | Qwen3 causal checkpoint |
| `SYN_DEVICE` / `SYN_DTYPE` | `auto` | `cpu`/`mps`/`cuda`; `float32`/`bfloat16`/`float16` |
| `SYN_READOUT` | `letters` | `letters`, `pmi`, `head`, `pointer` |
| `SYN_PMI_LIST_OPTIONS` | `false` | pmi: name options in the prefix; helps some items, leaks order |
| `SYN_HEAD_PATH` | unset | head: `.safetensors` checkpoint from `syn train-head` (its fitted temperature applies) |
| `SYN_POINTER_PATH` | unset | pointer: `pointer.safetensors` from `syn train-pointer`; set `SYN_MODEL` to that run's `backbone/` |
| `SYN_PROMPT_FORMAT` | `json` | `json` or `text` |
| `SYN_ORDERINGS` | `0` | Cyclic orderings per request; `0` = one per option |
| `SYN_TEMPERATURE` | `1` | Option-distribution temperature |
| `SYN_ABSTAIN_THRESHOLD` / `SYN_MIN_CONFIDENCE` / `SYN_MIN_ORDERING_AGREEMENT` | `0` | Abstention gates |
| `SYN_API_KEY` | unset | Bearer token on every classification route; `/health` stays open |
| `SYN_LOG_PATH` | unset | Optional JSONL decision log |
| `SYN_SGLANG_URL` | `http://127.0.0.1:30000` | SGLang server URL |
| `SYN_SGLANG_CHECK_MODEL` | `true` | Refuse to start if SGLang serves another model |
| `SYN_MAX_PROMPT_TOKENS` | `8192` | Reject longer prompts; never truncate |
| `SYN_LOCAL_BATCH_TOKENS` | `16384` | Token budget per batched local forward |

## Checks

```sh
HF_HOME=.cache/huggingface uv run --extra local pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
cd deploy/cloudflare && npm ci && npm test && npm run check
```
