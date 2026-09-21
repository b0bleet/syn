"""GET shorthand: /<label,label,...>/<text> for one-line classification.

    GET /spam,not+spam/Win+a+free+iPhone
    GET /?labels=spam,not+spam&text=Win+a+free+iPhone

Labels are comma separated. `+` means a space and percent escapes are decoded, so in the path
form a label or text containing a literal comma, plus, or slash is written `%2C`, `%2B`, `%2F`.
The query form decodes before splitting, so its labels cannot contain commas.
"""

from urllib.parse import unquote_plus

# Whole URLs must stay inside common server and proxy limits; long states use POST /v1/score.
MAX_TEXT_CHARS = 4000
MAX_LABEL_CHARS = 200
# First segments that belong to the real API. Without this, GET /v1/scores would be read as a
# classification into the single label "v1" and report a confusing validation error.
RESERVED = frozenset(
    {"v1", "health", "docs", "redoc", "openapi.json", "metrics", "favicon.ico", "robots.txt"}
)
DEFAULT_QUESTION = "Which label best applies to the text?"


class ShorthandError(ValueError):
    pass


def raw_path(scope: dict) -> str:
    """The path exactly as sent, before the server percent-decodes it.

    Starlette hands handlers a decoded path, so `%2C` would already be a comma and would split
    a label in half. ASGI servers expose the original bytes as `raw_path`.
    """
    raw = scope.get("raw_path")
    if raw is None:
        return scope.get("path", "")
    text = raw.decode("utf-8", "replace")
    # Some servers include the query string in raw_path.
    return text.split("?", 1)[0]


def parse(path: str) -> tuple[list[str], str]:
    """Split a raw request path into labels and the text to classify."""
    head, separator, tail = path.lstrip("/").partition("/")
    if not separator:
        raise ShorthandError(
            "Expected /<label,label,...>/<text>, for example /spam,not+spam/Win+a+free+iPhone"
        )
    if head.split(",")[0].lower() in RESERVED:
        raise ShorthandError(f"{head!r} is a reserved path prefix, not a label list")
    # Split before decoding, so a percent-escaped comma stays inside its label.
    return check([unquote_plus(part) for part in head.split(",")], unquote_plus(tail))


def check(labels: list[str], text: str) -> tuple[list[str], str]:
    """Strip and validate decoded labels and text; shared by the path and query forms."""
    labels = [label.strip() for label in labels]
    text = text.strip()
    if any(not label for label in labels):
        raise ShorthandError("Every label must be non-empty")
    if not 2 <= len(labels) <= 26:
        raise ShorthandError(f"Need 2 to 26 labels, got {len(labels)}")
    if len(labels) != len(set(labels)):
        raise ShorthandError("Labels must be unique")
    if any(len(label) > MAX_LABEL_CHARS for label in labels):
        raise ShorthandError(f"Each label is limited to {MAX_LABEL_CHARS} characters")
    if not text:
        raise ShorthandError("The text to classify is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ShorthandError(
            f"Text is {len(text)} characters; the URL form is limited to {MAX_TEXT_CHARS}. "
            "Use POST / or POST /v1/score for longer texts."
        )
    return labels, text


def build_request(labels: list[str], text: str, question: str | None = None) -> dict:
    """A ScoreRequest payload where each label is both the option id and its description."""
    return {
        "context": text,
        "question": (question or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION,
        "options": [{"id": label, "text": label} for label in labels],
    }
