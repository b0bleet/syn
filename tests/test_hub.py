import json
import random

import pytest

from syn.hub import _columns, convert, label_text, make_row
from syn.schema import EvalExample


class FakeFeature:
    def __init__(self, names):
        self.names = names


class FakeSplit:
    """Enough of a datasets.Dataset for the converter: features, len, iteration, shuffle, select."""

    def __init__(self, rows, names):
        self.rows = rows
        self.features = {"text": None, "label": FakeFeature(names)}

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed=0):
        shuffled = list(self.rows)
        random.Random(seed).shuffle(shuffled)
        return FakeSplit(shuffled, self.features["label"].names)

    def select(self, indices):
        return FakeSplit([self.rows[i] for i in indices], self.features["label"].names)


def fake_dataset(n_train=200, n_test=60, classes=("World", "Sports", "Business", "Sci/Tech")):
    def rows(n, offset):
        return [
            {
                "text": f"article number {i + offset} about {classes[i % len(classes)]}",
                "label": i % len(classes),
            }
            for i in range(n)
        ]

    return {
        "train": FakeSplit(rows(n_train, 0), list(classes)),
        "test": FakeSplit(rows(n_test, 10_000), list(classes)),
    }


@pytest.fixture
def patched(monkeypatch):
    holder = {"data": fake_dataset()}

    def load_dataset(name, revision=None):
        holder["called"] = (name, revision)
        return holder["data"]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        type("m", (), {"load_dataset": staticmethod(load_dataset)}),
    )
    return holder


def test_label_text_uses_gloss_then_falls_back():
    assert "World news" in label_text("fancyzhx/ag_news", "World")
    assert label_text("other/ds", "balance_not_updated") == "balance not updated"
    assert label_text("other/ds", "plain") == "plain"


def test_columns_detection_and_errors():
    assert _columns({"text": None, "label": FakeFeature(["a", "b"])}) == (
        "text",
        "label",
        ["a", "b"],
    )
    assert _columns({"sentence": None, "label": FakeFeature(["a"])})[0] == "sentence"
    with pytest.raises(ValueError, match="No label column"):
        _columns({"text": None, "other": None})
    with pytest.raises(ValueError, match="no class names"):
        _columns({"text": None, "label": None})


def test_make_row_samples_options_and_always_includes_the_answer():
    names = [f"c{i}" for i in range(10)]
    rng = random.Random(0)
    for _ in range(40):
        row = make_row("d", "q?", "some text", "c3", names, rng, 2, 5, all_options=False)
        ids = [o["id"] for o in row["request"]["options"]]
        assert "c3" in ids and row["expected_option_id"] == "c3"
        assert 2 <= len(ids) <= 5 and len(set(ids)) == len(ids)
    every = make_row("d", "q?", "t", "c3", names, rng, 2, 5, all_options=True)
    assert sorted(o["id"] for o in every["request"]["options"]) == sorted(names)


def test_convert_writes_valid_splits(patched, tmp_path):
    meta = convert("fancyzhx/ag_news", tmp_path, train=40, validation=10, test=20, seed=3)
    assert (meta["train"], meta["validation"], meta["test"]) == (40, 10, 20)
    assert meta["classes"] == 4 and meta["text_column"] == "text"
    assert meta["question"] == "What is the topic of this news article?"
    assert patched["called"] == ("fancyzhx/ag_news", None)
    contexts = {}
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines()]
        for row in rows:
            example = EvalExample.model_validate(row)  # answer is always among the options
            assert example.expected_option_id in {o.id for o in example.request.options}
            assert "World news" in next(
                (o.text for o in example.request.options if o.id == "World"), "World news"
            )
        contexts[split] = {r["request"]["context"] for r in rows}
    # Validation is carved from train, so it must not overlap it; test comes from a separate pool.
    assert not contexts["train"] & contexts["validation"]
    assert not contexts["train"] & contexts["test"]
    assert json.loads((tmp_path / "meta.json").read_text())["seed"] == 3


def test_convert_refuses_to_overwrite(patched, tmp_path):
    convert("x/y", tmp_path, train=5, validation=2, test=5)
    with pytest.raises(FileExistsError):
        convert("x/y", tmp_path, train=5, validation=2, test=5)


def test_convert_drops_unusable_rows(patched, tmp_path):
    """Every train row is unusable except four, so the loop must visit and reject all of them."""
    bad = fake_dataset(n_train=24, n_test=8)
    for i, row in enumerate(bad["train"].rows):
        if i % 6 == 0:
            continue
        row["text"] = None if i % 2 else "   "
    patched["data"] = bad
    # Ask for more rows than exist, so the loop cannot stop early and skips nothing silently.
    meta = convert("x/y", tmp_path, train=999, validation=1, test=8)
    assert meta["skipped_rows"] >= 15
    assert meta["train"] <= 4 and meta["train"] >= 1
    assert meta["question"] == "Which label best applies to the text?"
    for line in (tmp_path / "train.jsonl").read_text().splitlines():
        assert json.loads(line)["request"]["context"].strip()


def test_convert_rejects_out_of_range_labels(patched, tmp_path):
    bad = fake_dataset(n_train=12, n_test=8)
    for row in bad["train"].rows:
        row["label"] = 99
    patched["data"] = bad
    meta = convert("x/y", tmp_path, train=999, validation=1, test=8)
    assert meta["train"] == 0 and meta["skipped_rows"] >= 9


def test_convert_rejects_option_counts_the_schema_would_refuse(patched, tmp_path):
    with pytest.raises(ValueError, match="min_options"):
        convert("x/y", tmp_path / "a", train=2, validation=1, test=2, min_options=1)
    with pytest.raises(ValueError, match="max_options"):
        convert("x/y", tmp_path / "b", train=2, validation=1, test=2, max_options=30)
    with pytest.raises(ValueError, match="min_options"):
        convert("x/y", tmp_path / "c", train=2, validation=1, test=2, min_options=5, max_options=3)
    many = fake_dataset(n_train=60, n_test=30, classes=tuple(f"c{i}" for i in range(27)))
    patched["data"] = many
    with pytest.raises(ValueError, match="27 classes"):
        convert("x/y", tmp_path / "d", train=2, validation=1, test=2, all_options=True)
    # 26 classes is the ceiling and must still work.
    patched["data"] = fake_dataset(n_train=60, n_test=30, classes=tuple(f"c{i}" for i in range(26)))
    meta = convert("x/y", tmp_path / "e", train=5, validation=2, test=5, all_options=True)
    assert meta["test"] == 5
    for line in (tmp_path / "e" / "test.jsonl").read_text().splitlines():
        assert len(json.loads(line)["request"]["options"]) == 26


def test_convert_never_lets_a_train_text_reach_test(patched, tmp_path):
    data = fake_dataset(n_train=40, n_test=20)
    # Make the test split a verbatim copy of twenty training texts, as real datasets sometimes do.
    data["test"].rows = [dict(r) for r in data["train"].rows[:20]]
    patched["data"] = data
    # validation=10 consumes every carved row, so all 40 training texts end up written and every
    # one of the 20 copies collides with an emitted row.
    meta = convert("x/y", tmp_path, train=40, validation=10, test=20)
    assert meta["cross_split_duplicates_dropped"] == 20
    assert meta["test"] == 0
    # The guarantee is about the files written: a dropped duplicate always matched an emitted row.
    assert meta["cross_split_duplicates_dropped"] + meta["test"] == 20
    train = {
        json.loads(l)["request"]["context"]
        for l in (tmp_path / "train.jsonl").read_text().splitlines()
    }
    val = {
        json.loads(l)["request"]["context"]
        for l in (tmp_path / "validation.jsonl").read_text().splitlines()
    }
    assert not train & val


def test_convert_needs_a_train_split(patched, tmp_path):
    patched["data"] = {"test": fake_dataset()["test"]}
    with pytest.raises(ValueError, match="no train split"):
        convert("x/y", tmp_path, train=2, validation=1, test=2)
