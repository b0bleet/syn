"""Train the general head on a GPU machine, end to end.

    python deploy/runpod/train_job.py

Written for a RunPod pod started by scripts/runpod_train.py, but any machine with a GPU and
the [local,hub] extras installed works. With SYN_TRAIN_HF_REPO set (and HF_TOKEN), a Hugging
Face repo is the store and the pod needs no volume: data and cached features are pulled from
it first, new features are pushed as soon as they are cached, and the run directory and log
are pushed at the end (layout in src/syn/artifacts.py). Every step is skipped when its output
already exists, so a rerun picks up where the last one stopped:

1. data/: pull data/ from the repo (your own rows, imported suites), rebuild the public
   datasets (scripts/download_data.py), and import any suites named in SYN_TRAIN_SUITES.
2. features: pull features/<model>/ from the repo, cache the missing files for every
   source's train, validation, calibration, and test splits under
   $SYN_TRAIN_ROOT/features/<model>/, and push the new ones.
3. transfer: train the head on all sources and once without each source, with calibration in
   the checkpoint, the ordinal loss, and none/distractor augmentation, under
   $SYN_TRAIN_ROOT/heads/<model>/<run>/. RESULT.md there is the summary; it is also printed.
   The run directory and the log go to heads/<model>/<run>/ and logs/ in the repo.
   SYN_TRAIN_TASK=pointer skips the feature cache and instead adapts the backbone
   (rank-16 adapters, merged on save) and trains the pointer head on pointer-data/
   from the same store. The run lands at pointers/<model>/<run>/, including the merged
   backbone, and a held-out file pointer-data/transfer-dev.jsonl is scored at the end.
4. Stop the pod on success (RUNPOD_POD_ID and RUNPOD_API_KEY present, SYN_TRAIN_STOP_POD
   not 0). On failure the pod stays up so its logs can be read.

Backbone settings are the usual SYN_MODEL, SYN_REVISION, SYN_DTYPE. Job settings:
SYN_TRAIN_HF_REPO (<user>/<repo>, needs HF_TOKEN), SYN_TRAIN_ROOT (default /runpod-volume
when mounted, else /workspace), SYN_TRAIN_SOURCES (comma-separated data/ directories; default
every one with train, validation, and test), SYN_TRAIN_SUITES, SYN_TRAIN_RUN (default a
timestamp), SYN_TRAIN_LIMIT_PER_SOURCE (2000), SYN_TRAIN_EPOCHS (8), SYN_TRAIN_RANK (256),
SYN_TRAIN_LR (5e-4), SYN_TRAIN_SEED (7), SYN_TRAIN_P_NONE (0.1), SYN_TRAIN_P_NONE_DISTRACT
(0.12), SYN_TRAIN_P_DISTRACT (0.15), SYN_TRAIN_ORDINAL_WEIGHT (1.0).
Pointer task: SYN_TRAIN_POINTER_RANK (16), SYN_TRAIN_POINTER_EPOCHS (2, or --epochs),
SYN_TRAIN_POINTER_LR (5e-5), SYN_TRAIN_POINTER_BATCH (2), SYN_TRAIN_POINTER_ACCUMULATE (4),
SYN_TRAIN_POINTER_MAX_TOKENS (1024), SYN_TRAIN_POINTER_CHECKPOINTING (1),
SYN_TRAIN_ANCHOR (0), SYN_TRAIN_ANCHOR_WEIGHT (1), SYN_TRAIN_ANCHOR_ORDERINGS (1).
The anchor is off because its teacher is the letters readout in the chat prompt, while the
pointer learns a cloze prefix. `--anchor` sets SYN_TRAIN_ANCHOR=1.
SYN_TRAIN_ORDINAL_WEIGHT defaults to 0 for the pointer task (cross-entropy only).
SYN_TRAIN_BALANCE_SOURCES (0) gives every source the same total loss weight when enabled.
SYN_TRAIN_HOLDOUT_SELECTION (0) holds out the median-sized source and selects the checkpoint
on that source's validation rows. The locked transfer file is not used for selection.
SYN_TRAIN_POLICY_CASES (0) adds a copy that states the day count between dates when enabled.
Automatic uniform-label copies are disabled: removing a numeral does not prove uncertainty.
Rows come from pointer-data/<source>/{train,validation,calibration}.jsonl, not from data/.
test.jsonl beside those splits is scored and not trained on. With SYN_TRAIN_ANCHOR=1 the
base model's letters readout writes an anchor file and the pointer loss stays near it.
Serve with SYN_MODEL=hf://<repo>/pointers/<model>/<run>/backbone and
SYN_POINTER_PATH=hf://<repo>/pointers/<model>/<run>/pointer.safetensors.

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


def store() -> str | None:
    """The Hugging Face repo that holds data, features, and heads, when configured."""
    repo = setting("HF_REPO", None)
    if repo and not os.environ.get("HF_TOKEN"):
        raise SystemExit("SYN_TRAIN_HF_REPO needs HF_TOKEN (a write token) in the environment")
    return repo


def build_data(repo: str | None, log) -> None:
    if repo:
        from syn.artifacts import pull

        fetched = pull(repo, "data", REPO)
        log(f"pulled {len(fetched)} data files from {repo}")
    if not has_rows(DATA / "agnews" / "train.jsonl"):
        log("rebuilding the public datasets under data/")
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


def pointer_rows(root: Path) -> dict[str, list[Path]]:
    """Train, validation, and calibration JSONL files for the pointer task.

    `root` is the pointer-data directory. SYN_TRAIN_SOURCES limits which subdirectories
    are read. A directory counts when it has non-empty train and validation files.
    """
    wanted = setting("SOURCES", None, names)
    dirs = (
        [root / name for name in wanted]
        if wanted
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    files: dict[str, list[Path]] = {"train": [], "validation": [], "calibration": []}
    for directory in dirs:
        if not directory.is_dir():
            raise SystemExit(f"Missing source directory {directory}")
        if not (has_rows(directory / "train.jsonl") and has_rows(directory / "validation.jsonl")):
            continue
        for split in ("train", "validation", "calibration"):
            path = directory / f"{split}.jsonl"
            if has_rows(path):
                files[split].append(path)
    if not files["train"]:
        raise SystemExit(f"No pointer training rows under {root}")
    return files


def prepare_pointer_data(repo: str | None, log) -> tuple[Path, Path | None]:
    """Pull pointer-data/ from the store. Returns that directory and the held-out file, if any."""
    root = REPO / "pointer-data"
    if repo:
        from syn.artifacts import pull

        fetched = pull(repo, "pointer-data", REPO)
        log(f"pulled {len(fetched)} pointer-data files from {repo}")
    if not root.is_dir():
        raise SystemExit("Pointer training needs pointer-data/ in the store or on disk")
    held_out = root / "transfer-dev.jsonl"
    return root, held_out if has_rows(held_out) else None


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


def cache_features(
    dirs: list[Path], root: Path, repo: str | None, model_slug: str, log
) -> dict[str, list[Path]]:
    """Feature files per split; extracts the missing ones with one loaded backbone."""
    if repo:
        from syn.artifacts import pull

        fetched = pull(repo, f"features/{model_slug}", root.parent.parent)
        log(f"pulled {len(fetched)} feature files from {repo}")
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
    cached = 0
    try:
        for directory, split, dataset, out in pending:
            examples = [EvalExample.model_validate(row) for row in read_jsonl(dataset)]
            meta = {
                **base,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
            result = extract_with(backend, builder, examples, out, meta, source=directory.name)
            cached += 1
            log(f"{directory.name}/{split}: {result['n']} rows in {result['seconds']} s -> {out}")
    finally:
        # Whatever was cached survives the pod, even when a later file fails.
        if repo and cached:
            from syn.artifacts import push

            url = push(repo, root, f"features/{model_slug}", f"Cache {cached} feature files")
            log(f"pushed {cached} feature files to {repo}: {url}")
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


def summary(report: dict, model: str, run_dir: Path, serve_path: str) -> str:
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
        f"Serve it with `SYN_READOUT=head SYN_HEAD_PATH={serve_path}`.",
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


def write_anchors(train_files: list[Path], out: Path, limit_per_source: int, log) -> Path | None:
    """Score the training rows with the frozen base model and write an anchor file.

    One letters ordering per row. The file matches what `syn train-pointer --anchor` reads.
    Returns None when anchoring is off or no row could be scored. The model is dropped
    before the caller loads it again for training.
    """
    if setting("ANCHOR", "0") == "0":
        log("anchors off")
        return None
    import gc

    from syn.backends import LocalBackend, resolve_config, revision_commit
    from syn.config import Settings
    from syn.evaluation import example_digest
    from syn.pointer import load_rows
    from syn.prompt import PromptBuilder
    from syn.scoring import Scorer

    settings = Settings(readout="letters", orderings=setting("ANCHOR_ORDERINGS", 1, int))
    rows = load_rows(train_files, limit_per_source)
    config = resolve_config(settings)
    from transformers import AutoTokenizer

    from syn.artifacts import pretrained_call

    model_id, extra = pretrained_call(settings.model, settings.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **extra)
    builder = PromptBuilder(tokenizer, settings.max_prompt_tokens, settings.prompt_format)
    log(f"scoring {len(rows)} anchor rows with {settings.model}")
    backend = LocalBackend(settings, config)
    scorer = Scorer(settings, builder, backend, revision_commit(config))
    out.parent.mkdir(parents=True, exist_ok=True)
    scored = 0
    try:
        with out.open("w") as handle:
            for example in rows:
                try:
                    response = scorer.score(example.request)
                except (ValueError, RuntimeError) as exc:
                    log(f"anchor skip: {exc}")
                    continue
                handle.write(
                    json.dumps(
                        {
                            "example_sha256": example_digest(example),
                            "response": response.model_dump(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                scored += 1
                if scored % 500 == 0:
                    log(f"anchors {scored}/{len(rows)}")
    finally:
        backend.close()
        del backend, scorer, tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if not scored:
        log("no anchor rows scored")
        return None
    log(f"anchors {scored}/{len(rows)} -> {out}")
    return out


def run_pointer(repo: str | None, root: Path, model: str, model_slug: str, run: str, log) -> None:
    """Adapt the backbone and train the pointer head on pointer-data/, then score the held-out file."""
    from syn.pointer import train_pointer
    from syn.training import Augment

    data_root, held_out = prepare_pointer_data(repo, log)
    files = pointer_rows(data_root)
    log(
        "pointer rows: "
        + ", ".join(f"{split} {len(paths)}" for split, paths in files.items() if paths)
    )
    run_dir = root / "pointers" / model_slug / run
    limit = setting("LIMIT_PER_SOURCE", 0, int)
    anchor = write_anchors(files["train"], root / "anchors" / f"{run}.jsonl", limit, log)
    indomain = [
        path.with_name("test.jsonl")
        for path in files["train"]
        if has_rows(path.with_name("test.jsonl"))
    ]
    report = train_pointer(
        files["train"],
        files["validation"],
        run_dir,
        calibration=files["calibration"] or None,
        test=held_out,
        indomain=indomain or None,
        anchor=anchor,
        anchor_weight=setting("ANCHOR_WEIGHT", 1.0, float) if anchor else 0.0,
        rank=setting("POINTER_RANK", 16, int),
        head_dim=setting("POINTER_HEAD_DIM", 256, int),
        epochs=setting("POINTER_EPOCHS", setting("EPOCHS", 2, int), int),
        lr=setting("POINTER_LR", 5e-5, float),
        batch_size=setting("POINTER_BATCH", 2, int),
        accumulate=setting("POINTER_ACCUMULATE", 4, int),
        weight_decay=setting("POINTER_WEIGHT_DECAY", 0.01, float),
        seed=setting("SEED", 7, int),
        augment=Augment(
            setting("P_NONE", 0.1, float),
            setting("P_NONE_DISTRACT", 0.12, float),
            setting("P_DISTRACT", 0.15, float),
        ),
        ordinal_weight=setting("ORDINAL_WEIGHT", 0.0, float),
        max_tokens=setting("POINTER_MAX_TOKENS", 1024, int),
        limit_per_source=limit,
        balance_sources=setting("BALANCE_SOURCES", "0") != "0",
        holdout_selection=setting("HOLDOUT_SELECTION", "0") != "0",
        policy_cases=setting("POLICY_CASES", "0") != "0",
        checkpointing=setting("POINTER_CHECKPOINTING", "1") != "0",
        log=log,
    )
    lines = [
        f"# Pointer readout on {model}",
        "",
        f"Validation top-1 {report['best_val_top1']:.3f}. Temperature {report['temperature']}.",
    ]
    if report.get("held_out_top1") is not None:
        lines.append(f"Held-out top-1 {report['held_out_top1']:.3f} on `{held_out}`.")
    if report.get("in_domain_top1") is not None:
        lines.append(f"In-domain test top-1 {report['in_domain_top1']:.3f}.")
    if repo:
        serve = (
            f"SYN_MODEL=hf://{repo}/pointers/{model_slug}/{run}/backbone "
            f"SYN_READOUT=pointer "
            f"SYN_POINTER_PATH=hf://{repo}/pointers/{model_slug}/{run}/pointer.safetensors"
        )
        lines += [
            "",
            f"Serve with `{serve}`.",
            f"The run is in the store at `pointers/{model_slug}/{run}/`.",
        ]
    else:
        serve = (
            f"SYN_MODEL={run_dir / 'backbone'} SYN_READOUT=pointer "
            f"SYN_POINTER_PATH={run_dir / 'pointer.safetensors'}"
        )
        lines += ["", f"Serve with `{serve}`."]
    text = "\n".join(lines) + "\n"
    (run_dir / "RESULT.md").write_text(text)
    print(text, flush=True)
    if repo:
        from syn.artifacts import push

        url = push(
            repo, run_dir, f"pointers/{model_slug}/{run}", f"Add pointer run {run} on {model}"
        )
        log(f"pushed the pointer run to {repo}: {url}")


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
    model_slug = slug(model)
    repo = store()
    root = Path(
        setting("ROOT", "/runpod-volume" if Path("/runpod-volume").is_dir() else "/workspace")
    )
    run = setting("RUN", datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    features_dir = root / "features" / model_slug
    run_dir = root / "heads" / model_slug / run
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = (logs / f"{run}.log").open("a")

    def log(message: str) -> None:
        line = f"[{time.time() - started:7.0f}s] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    mode = setting("TASK", "head")
    if mode not in ("head", "pointer"):
        raise SystemExit(f"SYN_TRAIN_TASK must be head or pointer, got {mode!r}")
    log(f"model {model}, root {root}, run {run}, task {mode}, store {repo or 'local only'}")
    if mode == "pointer":
        run_pointer(repo, root, model, model_slug, run, log)
        log("done")
        log_file.flush()
        if repo:
            from syn.artifacts import push

            push(repo, logs, "logs", f"Add log {run}")
        stop_pod(log)
        return
    build_data(repo, log)
    dirs = sources()
    if len(dirs) < 2:
        raise SystemExit(f"Need at least two sources with train, validation, and test; got {dirs}")
    log(f"sources: {[d.name for d in dirs]}")
    features_dir.mkdir(parents=True, exist_ok=True)
    files = cache_features(dirs, features_dir, repo, model_slug, log)
    report = train(files, run_dir, log)
    head_in_repo = f"heads/{model_slug}/{run}/all-sources.safetensors"
    serve_path = f"hf://{repo}/{head_in_repo}" if repo else str(run_dir / "all-sources.safetensors")
    text = summary(report, model, run_dir, serve_path)
    (run_dir / "RESULT.md").write_text(text)
    print(text, flush=True)
    if repo:
        from syn.artifacts import push

        url = push(repo, run_dir, f"heads/{model_slug}/{run}", f"Add run {run} on {model}")
        log(f"pushed the run to {repo}: {url}")
        log_file.flush()
        push(repo, logs, "logs", f"Add log {run}")
    log("done")
    stop_pod(log)


if __name__ == "__main__":
    main()
