import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrained Qwen decision scorer")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve using SYN_* environment settings")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)

    evaluate = sub.add_parser(
        "evaluate",
        help="Score labeled JSONL through the service; resumes if the output file exists",
    )
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--url", default="http://127.0.0.1:8765")
    evaluate.add_argument("--shuffle-options", action="store_true")
    evaluate.add_argument("--seed", type=int, default=42)

    bench = sub.add_parser(
        "bench",
        help="Send the same labeled rows to System One endpoints such as Jev; resumes",
    )
    bench.add_argument("datasets", type=Path, nargs="+")
    bench.add_argument("--out", type=Path, required=True)
    bench.add_argument(
        "--target",
        action="append",
        required=True,
        help="NAME[@MODEL][=URL], repeatable; the first is the reference. "
        "jev needs TYPESAFE_API_KEY; other targets send <NAME>_API_KEY when set",
    )
    bench.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Requests in flight per target; above 1, latency includes queueing at the target",
    )
    bench.add_argument("--limit", type=int, default=0, help="Only the first N rows per dataset")

    calibrate = sub.add_parser("calibrate", help="Fit temperature on calibration predictions")
    calibrate.add_argument("predictions", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser(
        "compare", help="Paired accuracy comparison of two evaluation outputs on one dataset"
    )
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--samples", type=int, default=1000)

    synthetic = sub.add_parser(
        "synthetic", help="Generate the synthetic routing dataset for pipeline validation"
    )
    synthetic.add_argument("--out", type=Path, required=True)
    synthetic.add_argument("--train", type=int, default=2000)
    synthetic.add_argument("--validation", type=int, default=400)
    synthetic.add_argument("--test", type=int, default=400)
    synthetic.add_argument("--seed", type=int, default=7)

    import_hf = sub.add_parser(
        "import-hf", help="Convert a Hugging Face classification dataset into decision rows"
    )
    import_hf.add_argument("dataset", help="e.g. fancyzhx/ag_news")
    import_hf.add_argument("--out", type=Path, required=True)
    import_hf.add_argument("--train", type=int, default=4000)
    import_hf.add_argument("--validation", type=int, default=500)
    import_hf.add_argument("--test", type=int, default=1000)
    import_hf.add_argument("--seed", type=int, default=7)
    import_hf.add_argument("--min-options", type=int, default=2)
    import_hf.add_argument("--max-options", type=int, default=6)
    import_hf.add_argument(
        "--all-options",
        action="store_true",
        help="Show every class on every row, for benchmark comparability",
    )
    import_hf.add_argument("--question")
    import_hf.add_argument("--revision")

    features = sub.add_parser(
        "features", help="Cache frozen-backbone features for a labeled JSONL (SYN_* model settings)"
    )
    features.add_argument("dataset", type=Path)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--limit", type=int, default=0)

    train = sub.add_parser("train-head", help="Train the AttentionHead on cached features")
    train.add_argument("train", type=Path)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--rank", type=int, default=256)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=7)

    eval_head = sub.add_parser(
        "eval-head", help="Evaluate a head on cached features, with the shuffled-context control"
    )
    eval_head.add_argument("head", type=Path)
    eval_head.add_argument("features", type=Path)
    eval_head.add_argument("--batch-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("syn.api:create_app", factory=True, host=args.host, port=args.port, workers=1)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(args.dataset, args.output, args.url, args.shuffle_options, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "bench":
        from .bench import TargetError, bench, parse_target

        try:
            targets = [parse_target(spec) for spec in args.target]
        except ValueError as exc:
            parser.error(str(exc))
        try:
            result = bench(args.datasets, args.out, targets, args.concurrency, args.limit)
        except TargetError as exc:
            raise SystemExit(f"syn bench: {exc}") from exc
        print(json.dumps(result, indent=2))
    elif args.command == "calibrate":
        from .evaluation import fit_temperature

        result = fit_temperature(args.predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(result, out, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        from .evaluation import compare

        print(json.dumps(compare(args.run_a, args.run_b, args.samples), indent=2))
    elif args.command == "synthetic":
        from .synthetic import generate

        print(json.dumps(generate(args.out, args.train, args.validation, args.test, args.seed)))
    elif args.command == "import-hf":
        from .hub import convert

        result = convert(
            args.dataset,
            args.out,
            train=args.train,
            validation=args.validation,
            test=args.test,
            seed=args.seed,
            min_options=args.min_options,
            max_options=args.max_options,
            all_options=args.all_options,
            question=args.question,
            revision=args.revision,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "class_names"}, indent=2))
    elif args.command == "features":
        from .config import Settings
        from .features import extract_dataset

        print(json.dumps(extract_dataset(Settings(), args.dataset, args.out, args.limit), indent=2))
    elif args.command == "train-head":
        from .training import train_head

        result = train_head(
            args.train,
            args.validation,
            args.out,
            rank=args.rank,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
    else:
        from .training import evaluate_head

        print(json.dumps(evaluate_head(args.head, args.features, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
