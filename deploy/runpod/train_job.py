"""Train the general head on a GPU machine, end to end.

    python deploy/runpod/train_job.py

Written for a RunPod pod started by scripts/runpod_train.py, but any machine with a GPU and
the [local,hub] extras installed works. Every step is skipped when its output already exists,
so a rerun after a failure picks up where it stopped:

1. data/: rebuild the public datasets (scripts/download_data.py) and import any System One
   suites named in SYN_TRAIN_SUITES (directories of labelled requests, e.g. on the volume).
2. features: cache frozen-backbone features for every source's train, validation,
   calibration, and test splits under $SYN_TRAIN_ROOT/features/<model>/.
3. transfer: train the head on all sources and once without each source, with calibration in
   the checkpoint, the ordinal loss, and none/distractor augmentation, under
   $SYN_TRAIN_ROOT/heads/<model>/<run>/. RESULT.md there is the summary; it is also printed.
4. Optionally upload the run directory to a Hugging Face repo (HF_TOKEN and
   SYN_TRAIN_UPLOAD_REPO), then stop the pod on success (RUNPOD_POD_ID and RUNPOD_API_KEY
   present, SYN_TRAIN_STOP_POD not 0). On failure the pod stays up so its logs can be read.

Backbone settings are the usual SYN_MODEL, SYN_REVISION, SYN_DTYPE. Job settings:
SYN_TRAIN_ROOT (default /runpod-volume when mounted, else /workspace), SYN_TRAIN_SOURCES
(comma-separated data/ directories; default every one with train, validation, and test),
SYN_TRAIN_SUITES, SYN_TRAIN_RUN (default a timestamp), SYN_TRAIN_LIMIT_PER_SOURCE (2000),
SYN_TRAIN_EPOCHS (8), SYN_TRAIN_RANK (256), SYN_TRAIN_LR (5e-4), SYN_TRAIN_SEED (7),
SYN_TRAIN_P_NONE (0.1), SYN_TRAIN_P_NONE_DISTRACT (0.12), SYN_TRAIN_P_DISTRACT (0.15),
SYN_TRAIN_ORDINAL_WEIGHT (1.0).

Memory: features are float16 and stay in RAM while heads train. At hidden size 4096 that is
about one megabyte per row, so 2000 rows per source over six sources needs around 20 GB.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SPLITS = ("train", "validation", "calibration", "test")
RUNPOD_API = "https://rest.runpod.io/v1"


def setting(name: str, default, cast=str):
    value = os.environ.get(f"SYN_TRAIN_{name}")
    return default if value in (None, "") else cast(value)


def names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", model)


def has_rows(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def build_data(log) -> None:
    if not has_rows(DATA / "agnews" / "train.jsonl"):
        log("rebuilding data/ from Hugging Face")
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "download_data.py")], check=True, cwd=REPO
        )
    for suite in setting("SUITES", [], names):
        path = Path(suite)
        out = DATA / path.name
        if out.exists():
            continue
        from syn.suites import convert

        log(f"importing {path} -> {out}")
        log(json.dumps(convert([path], out)))


def sources() -> list[Path]:
    wanted = setting("SOURCES", None, names)
    dirs = (
        [DATA / name for name in wanted]
        if wanted
        else sorted(path for path in DATA.iterdir() if path.is_dir())
    )
    return [
        path
        for path in dirs
        if all(has_rows(path / f"{split}.jsonl") for split in ("train", "validation", "test"))
    ]


def cache_features(dirs: list[Path], root: Path, log) -> dict[str, list[Path]]:
    """Feature files per split; extracts the missing ones with one loaded backbone."""
    files: dict[str, list[Path]] = {split: [] for split in SPLITS}
    pending = []
    for directory in dirs:
        for split in SPLITS:
            dataset = directory / f"{split}.jsonl"
            if not has_rows(dataset):
                continue
            out = root / f"{directory.name}-{split}.npz"
            files[split].append(out)
            if not out.exists():
                pending.append((directory, split, dataset, out))
    if not pending:
        log(f"all {sum(map(len, files.values()))} feature files present under {root}")
        return files
    from transformers import AutoTokenizer

    from syn.backends import LocalBackend, resolve_config, revision_commit
    from syn.config import Settings
    from syn.evaluation import read_jsonl
    from syn.features import extract_with
    from syn.prompt import PromptBuilder
    from syn.schema import EvalExample

    settings = Settings()
    config = resolve_config(settings)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model, revision=settings.revision, trust_remote_code=False
    )
    builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
    log(f"loading {settings.model} @ {settings.revision}")
    backend = LocalBackend(settings, config)
    base = {
        "model": settings.model,
        "revision": settings.revision,
        "revision_commit": revision_commit(config),
        "dtype": str(backend.dtype).replace("torch.", ""),
    }
    for directory, split, dataset, out in pending:
        examples = [EvalExample.model_validate(row) for row in read_jsonl(dataset)]
        meta = {
            **base,
            "dataset": str(dataset),
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        }
        result = extract_with(backend, builder, examples, out, meta, source=directory.name)
        log(f"{directory.name}/{split}: {result['n']} rows in {result['seconds']} s -> {out}")
    return files


def train(files: dict[str, list[Path]], out_dir: Path, log) -> dict:
    from syn.training import Augment, transfer

    augment = Augment(
        setting("P_NONE", 0.1, float),
        setting("P_NONE_DISTRACT", 0.12, float),
        setting("P_DISTRACT", 0.15, float),
    )
    return transfer(
        files["train"],
        files["validation"],
        files["test"],
        out_dir,
        calibration=files["calibration"] or None,
        limit_per_source=setting("LIMIT_PER_SOURCE", 2000, int),
        rank=setting("RANK", 256, int),
        epochs=setting("EPOCHS", 8, int),
        lr=setting("LR", 5e-4, float),
        seed=setting("SEED", 7, int),
        augment=augment,
        ordinal_weight=setting("ORDINAL_WEIGHT", 1.0, float),
        log=log,
    )


def summary(report: dict, model: str, run_dir: Path) -> str:
    general = report["general"]
    lines = [
        f"# General head on {model}",
        "",
        (
            f"Checkpoints in `{run_dir}`. All-sources head: validation top-1 "
            f"{general['best_val_top1']:.3f}, temperature {general['temperature']:.2f} "
            f"(fitted on {general['temperature_fit']['on']})."
        ),
        "",
        (
            "Transfer holds the source out of training entirely; in-domain is the all-sources "
            "head on the same test rows. Control is the held-out head with shuffled contexts."
        ),
        "",
        (
            "| source | test rows | in-domain top-1 | transfer top-1 | 95% CI | gap "
            "| transfer ECE | control |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for source, entry in report["sources"].items():
        if "skipped" in entry:
            lines.append(f"| {source} | | | | | | | skipped: {entry['skipped']} |")
            continue
        low, high = entry["transfer"]["top1_ci95"]
        lines.append(
            f"| {source} | {entry['test_examples']} | {entry['in_domain']['top1']:.3f} "
            f"| {entry['transfer']['top1']:.3f} | {low:.3f}-{high:.3f} "
            f"| {entry['gap_in_domain_minus_transfer']:+.3f} | {entry['transfer']['ece_10_bins']:.3f} "
            f"| {entry['control_top1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def upload(run_dir: Path, log) -> None:
    repo = setting("UPLOAD_REPO", None)
    token = os.environ.get("HF_TOKEN")
    if not repo or not token:
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, private=True)
    api.upload_folder(folder_path=str(run_dir), repo_id=repo, path_in_repo=run_dir.name)
    log(f"uploaded {run_dir} to {repo}/{run_dir.name}")


def stop_pod(log) -> None:
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not pod or not key or setting("STOP_POD", "1") == "0":
        return
    import httpx

    response = httpx.delete(
        f"{RUNPOD_API}/pods/{pod}", headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    log(f"stop pod {pod}: HTTP {response.status_code}")


def main() -> None:
    started = time.time()
    model = os.environ.get("SYN_MODEL", "Qwen/Qwen3-0.6B")
    root = Path(
        setting("ROOT", "/runpod-volume" if Path("/runpod-volume").is_dir() else "/workspace")
    )
    run = setting("RUN", datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    features_dir = root / "features" / slug(model)
    run_dir = root / "heads" / slug(model) / run
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = (logs / f"{run}.log").open("a")

    def log(message: str) -> None:
        line = f"[{time.time() - started:7.0f}s] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"model {model}, root {root}, run {run}")
    build_data(log)
    dirs = sources()
    if len(dirs) < 2:
        raise SystemExit(f"Need at least two sources with train, validation, and test; got {dirs}")
    log(f"sources: {[d.name for d in dirs]}")
    features_dir.mkdir(parents=True, exist_ok=True)
    files = cache_features(dirs, features_dir, log)
    report = train(files, run_dir, log)
    text = summary(report, model, run_dir)
    (run_dir / "RESULT.md").write_text(text)
    print(text, flush=True)
    upload(run_dir, log)
    log("done")
    stop_pod(log)


if __name__ == "__main__":
    main()
