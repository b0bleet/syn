"""Import System One labelled requests into decision rows.

A labelled request is the body `/v1/systemone` accepts, plus the answer to each question:

    {"state": ..., "questions": {"name": {"type": "noul" | "choice" | "score",
                                          "instructions": ..., "criteria": ...,
                                          "label": ..., "src": "task"}},
     "_meta": {"source": "dataset", ...}}

Every question with a label becomes one row, rendered exactly as `/v1/systemone` renders it:
the same state text and the same option texts, so a head trained on these rows sees at
training time what it will see when served. Score questions are marked ordinal. The source
tag is the question's `src`, else the record's `_meta.source`, else the file's directory name.

    syn import-systemone path/to/suite --out data/suite

A suite directory holds train, calibration, development, and test files; development becomes
validation.jsonl so the split names match the rest of data/. Single files work too.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .evaluation import read_jsonl
from .features import default_source
from .schema import EvalExample
from .systemone import ChoiceQuestion, NoulQuestion, Question, render, to_request

SPLITS = {
    "train": "train",
    "calibration": "calibration",
    "development": "validation",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
}
QUESTION_FIELDS = ("type", "instructions", "criteria")
QUESTION_ADAPTER = TypeAdapter(Question)


def expected_id(question, label) -> str | None:
    """The option id a label names, in this service's ids; None when it names nothing."""
    if isinstance(question, NoulQuestion):
        if isinstance(label, str):
            label = label.strip().lower()
            return {"true": "yes", "yes": "yes", "false": "no", "no": "no"}.get(label)
        return ("yes" if label else "no") if isinstance(label, bool) else None
    if isinstance(question, ChoiceQuestion):
        return str(label) if str(label) in question.criteria else None
    if isinstance(label, bool) or not isinstance(label, int):
        return None
    return str(label) if 0 <= label < len(question.criteria) else None


def convert_record(record, source: str) -> tuple[list[dict], Counter]:
    """Rows for every labelled question in one record, plus counts of what was skipped."""
    skipped: Counter = Counter()
    if (
        not isinstance(record, dict)
        or "state" not in record
        or not isinstance(record.get("questions"), dict)
    ):
        skipped["malformed_record"] += 1
        return [], skipped
    state = render(record["state"])
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    rows = []
    for item in record["questions"].values():
        if not isinstance(item, dict):
            skipped["malformed_question"] += 1
            continue
        if "label" not in item:
            skipped["unlabelled"] += 1
            continue
        try:
            question = QUESTION_ADAPTER.validate_python(
                {k: item[k] for k in QUESTION_FIELDS if k in item}
            )
        except ValidationError:
            skipped["invalid_question"] += 1
            continue
        expected = expected_id(question, item["label"])
        if expected is None:
            skipped["label_names_no_option"] += 1
            continue
        try:
            request = to_request(state, question)
        except ValueError:
            skipped["invalid_request"] += 1
            continue
        if request is None:
            skipped["single_option"] += 1
            continue
        row = {
            "request": request.model_dump(),
            "expected_option_id": expected,
            "ordinal": question.type == "score",
            "source": str(item.get("src") or meta.get("source") or source)[:128],
        }
        rows.append(EvalExample.model_validate(row).model_dump())
    return rows, skipped


def convert(paths: list[Path], out: Path, source: str | None = None) -> dict:
    """Write one JSONL per input file into `out`, named by split. Refuses to overwrite."""
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files += sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No .jsonl files to import")
    targets = {}
    for path in files:
        target = out / f"{SPLITS.get(path.stem, path.stem)}.jsonl"
        if target.exists() or target in targets.values():
            raise FileExistsError(target)
        targets[path] = target
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"out": str(out), "files": {}}
    for path, target in targets.items():
        rows, skipped = [], Counter()
        for record in read_jsonl(path):
            new, skip = convert_record(record, source or default_source(path))
            rows += new
            skipped += skip
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        report["files"][str(path)] = {
            "out": str(target),
            "rows": len(rows),
            "ordinal_rows": sum(r["ordinal"] for r in rows),
            "sources": sorted({r["source"] for r in rows}),
            "skipped": dict(skipped),
        }
    return report
