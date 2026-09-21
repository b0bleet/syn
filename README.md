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

GET shorthand for short texts (path is `/<label,label,...>/<text>`):

```sh
curl "http://127.0.0.1:8765/spam,ham/Win+a+free+iPhone?format=label"   # -> "spam"
curl "http://127.0.0.1:8765/billing,technical,sales/I+was+charged+twice?q=Which+team%3F"
```

`+` means a space; percent-encode `%2C`, `%2B`, `%2F` inside labels. Interactive API docs at `/docs`, health at `/health`. A GET puts the classified text in the URL, which lands in access logs, so don't send personal data through it.

## How it scores

- **letters** (default): builds N cyclic option orderings so every option occupies every position once, reads label log-probs, averages, softmaxes. `ordering_agreement` exposes ordering disagreement, the signal that catches confidently wrong answers.
- **pmi**: scores each option's text likelihood given the context, prior-corrected, in one masked forward. Order-invariant by construction. Helps with short label words; weak on long descriptive options.
- **head**: a trained cross-attention head over frozen backbone features (`syn train-head`); local backend only, needs `SYN_HEAD_PATH`.

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

Trained head (frozen features cached once, ~1M-param head trains in seconds):

```sh
uv run --extra local syn features data/train.jsonl --out runs/feats/train.npz  # + validation, test
uv run --extra local syn train-head runs/feats/train.npz \
  --validation runs/feats/validation.npz --out runs/head.safetensors
uv run --extra local syn eval-head runs/head.safetensors runs/feats/test.npz
SYN_READOUT=head SYN_HEAD_PATH=runs/head.safetensors uv run --extra local syn serve
```

`eval-head` reports a shuffled-context control alongside real accuracy; if the control doesn't collapse toward chance, the head is reading option priors, not the state.

## SGLang backend

```sh
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000
SYN_BACKEND=sglang SYN_MODEL=Qwen/Qwen3-8B uv run syn serve
```

Startup refuses a remote running a different model or revision; `/health` probes it and returns 503 `degraded` when unreachable. RunPod image: `deploy/runpod/`.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `SYN_BACKEND` | `local` | `local` or `sglang` |
| `SYN_MODEL` | `Qwen/Qwen3-0.6B` | Qwen3 causal checkpoint |
| `SYN_DEVICE` / `SYN_DTYPE` | `auto` | `cpu`/`mps`/`cuda`; `float32`/`bfloat16`/`float16` |
| `SYN_READOUT` | `letters` | `letters`, `pmi`, `head` |
| `SYN_PMI_LIST_OPTIONS` | `false` | pmi: name options in the prefix; helps some items, leaks order |
| `SYN_HEAD_PATH` | unset | head: `.safetensors` checkpoint from `syn train-head` |
| `SYN_PROMPT_FORMAT` | `json` | `json` or `text` |
| `SYN_ORDERINGS` | `0` | Cyclic orderings per request; `0` = one per option |
| `SYN_TEMPERATURE` | `1` | Option-distribution temperature |
| `SYN_ABSTAIN_THRESHOLD` / `SYN_MIN_CONFIDENCE` / `SYN_MIN_ORDERING_AGREEMENT` | `0` | Abstention gates |
| `SYN_API_KEY` | unset | Bearer token on `/v1/score` and the GET shorthand; `/health` stays open |
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
```
