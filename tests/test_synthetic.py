import json

import pytest

from syn.schema import EvalExample
from syn.synthetic import DEPARTMENTS, generate


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_generate_is_valid_unique_and_deterministic(tmp_path):
    meta = generate(tmp_path / "a", train=30, validation=8, test=8, seed=3)
    assert (meta["train"], meta["validation"], meta["test"]) == (30, 8, 8)
    contexts = []
    for split in ("train", "validation", "test"):
        for row in rows(tmp_path / "a" / f"{split}.jsonl"):
            example = EvalExample.model_validate(row)  # label is always among the options
            assert example.expected_option_id in DEPARTMENTS
            assert 2 <= len(example.request.options) <= 6
            assert len({o.id for o in example.request.options}) == len(example.request.options)
            contexts.append(example.request.context)
    assert len(set(contexts)) == len(contexts)
    generate(tmp_path / "b", train=30, validation=8, test=8, seed=3)
    assert (tmp_path / "a" / "train.jsonl").read_text() == (
        tmp_path / "b" / "train.jsonl"
    ).read_text()
    generate(tmp_path / "c", train=30, validation=8, test=8, seed=4)
    assert (tmp_path / "a" / "train.jsonl").read_text() != (
        tmp_path / "c" / "train.jsonl"
    ).read_text()
    with pytest.raises(FileExistsError):
        generate(tmp_path / "a", train=1, validation=1, test=1)
