import json

import pytest
import uvicorn

from syn import cli


def test_cli_evaluate_and_calibrate(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_evaluate(*args):
        seen["evaluate"] = args
        return {"ok": 1}

    monkeypatch.setattr("syn.evaluation.evaluate", fake_evaluate)
    cli.main(
        ["evaluate", "data.jsonl", "--output", "out.jsonl", "--shuffle-options", "--seed", "7"]
    )
    dataset, output, url, shuffle, seed = seen["evaluate"]
    assert (str(dataset), str(output)) == ("data.jsonl", "out.jsonl")
    assert (url, shuffle, seed) == ("http://127.0.0.1:8765", True, 7)
    assert json.loads(capsys.readouterr().out) == {"ok": 1}

    monkeypatch.setattr("syn.evaluation.fit_temperature", lambda path: {"temperature": 2.0})
    output = tmp_path / "nested" / "temperature.json"
    cli.main(["calibrate", "predictions.jsonl", "--output", str(output)])
    assert json.loads(output.read_text()) == {"temperature": 2.0}
    with pytest.raises(FileExistsError):
        cli.main(["calibrate", "predictions.jsonl", "--output", str(output)])


def test_cli_compare(monkeypatch, capsys):
    seen = {}

    def fake_compare(a, b, samples):
        seen.update(a=str(a), b=str(b), samples=samples)
        return {"difference_b_minus_a": 0.1}

    monkeypatch.setattr("syn.evaluation.compare", fake_compare)
    cli.main(["compare", "a.jsonl", "b.jsonl", "--samples", "50"])
    assert seen == {"a": "a.jsonl", "b": "b.jsonl", "samples": 50}
    assert json.loads(capsys.readouterr().out) == {"difference_b_minus_a": 0.1}


def test_cli_training_commands(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        "syn.synthetic.generate",
        lambda out, train, validation, test, seed: (
            seen.setdefault("synthetic", (str(out), train, validation, test, seed)) or {"ok": 1}
        ),
    )
    cli.main(["synthetic", "--out", "data/syn", "--train", "10", "--seed", "3"])
    assert seen["synthetic"] == ("data/syn", 10, 400, 400, 3)

    monkeypatch.setattr(
        "syn.features.extract_dataset",
        lambda settings, dataset, out, limit: (
            seen.setdefault("features", (settings.model, str(dataset), str(out), limit)) or {"n": 1}
        ),
    )
    cli.main(["features", "d.jsonl", "--out", "f.npz", "--limit", "5"])
    assert seen["features"] == ("Qwen/Qwen3-0.6B", "d.jsonl", "f.npz", 5)

    def fake_train(train, validation, out, **kwargs):
        seen["train"] = (str(train), str(validation), str(out), kwargs)
        return {"best_val_top1": 1.0}

    monkeypatch.setattr("syn.training.train_head", fake_train)
    cli.main(["train-head", "t.npz", "--validation", "v.npz", "--out", "h.st", "--rank", "8"])
    assert seen["train"][:3] == ("t.npz", "v.npz", "h.st") and seen["train"][3]["rank"] == 8

    monkeypatch.setattr(
        "syn.training.evaluate_head",
        lambda head, features, batch_size: {"head": str(head), "bs": batch_size},
    )
    capsys.readouterr()
    cli.main(["eval-head", "h.st", "f.npz", "--batch-size", "16"])
    assert json.loads(capsys.readouterr().out) == {"head": "h.st", "bs": 16}


def test_cli_serve(monkeypatch):
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(kwargs, app=app))
    cli.main(["serve", "--port", "9000"])
    assert seen["app"] == "syn.api:create_app"
    assert seen["factory"] is True and seen["workers"] == 1
    assert (seen["host"], seen["port"]) == ("127.0.0.1", 9000)


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        cli.main([])
