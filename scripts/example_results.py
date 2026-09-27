"""Save the playground examples' results into deploy/cloudflare/public/index.html.

    SIFTY_API_KEY=sifty_... python3 scripts/example_results.py [https://sifty.dev]

Visitors who haven't registered see these saved results when they click an example, so the
page shows real output without calling the API for them. Run it again after a model change.
Examples live in the page's <script id="examples-data"> block; this fills each one's "result".
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "deploy" / "cloudflare" / "public" / "index.html"
BLOCK = re.compile(r'(<script id="examples-data" type="application/json">\n)(.*?)(\n</script>)', re.S)


def classify(base: str, key: str, example: dict) -> dict:
    payload = {"labels": [label.strip() for label in example["labels"].split(",")]}
    payload["image" if example.get("mode") == "image" else "input"] = example["input"]
    request = urllib.request.Request(
        base.rstrip("/") + "/",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)["results"][0]
    return {
        "label": result["label"],
        "confidence": round(result["confidence"], 4),
        "scores": {name: round(p, 4) for name, p in result["scores"].items()},
    }


def main() -> None:
    key = os.environ.get("SIFTY_API_KEY")
    if not key:
        sys.exit("Set SIFTY_API_KEY to a key from https://sifty.dev/account")
    base = sys.argv[1] if len(sys.argv) > 1 else "https://sifty.dev"
    page = PAGE.read_text()
    match = BLOCK.search(page)
    if not match:
        sys.exit(f'No <script id="examples-data"> block in {PAGE}')
    examples = json.loads(match.group(2))
    for example in examples:
        example["result"] = classify(base, key, example)
        print(f'{example["name"]}: {example["result"]["label"]} ({example["result"]["confidence"]:.2f})')
    body = "[\n" + ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in examples) + "\n]"
    PAGE.write_text(page[: match.start(2)] + body + page[match.end(2) :])
    print(f"Saved {len(examples)} results in {PAGE}")


if __name__ == "__main__":
    main()
