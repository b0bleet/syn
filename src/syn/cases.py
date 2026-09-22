"""Extra policy rows: state the day count, and train an even answer when the deciding fact is gone.

A policy row is a written rule plus a case. When the case gives two dates, a copy of the row
appends the number of days between them. A second copy drops the case sentence that carries
the number, and training then targets a uniform distribution over the options instead of the
original label.
"""

import re
from datetime import date

from .schema import EvalExample

_MONTHS = {
    name: index
    for index, name in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
_MONTHS.update({name[:3]: index for name, index in _MONTHS.items()})
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MDY = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2}),\s+(\d{4})\b",
    re.IGNORECASE,
)


def _month(name: str) -> int:
    key = name.lower()
    if key == "sept":
        key = "sep"
    return _MONTHS[key[:3]]


def date_facts(text: str) -> str:
    """One sentence per pair of absolute dates, later mention compared with each earlier one.

    An empty string means fewer than two dates. The wording of each date is kept as written.
    """
    found: list[tuple[int, str, date]] = []
    for match in _ISO.finditer(text):
        year, month, day = (int(match.group(i)) for i in (1, 2, 3))
        try:
            found.append((match.start(), match.group(0), date(year, month, day)))
        except ValueError:
            continue
    for match in _MDY.finditer(text):
        try:
            found.append(
                (
                    match.start(),
                    match.group(0),
                    date(int(match.group(3)), _month(match.group(1)), int(match.group(2))),
                )
            )
        except ValueError:
            continue
    found.sort(key=lambda item: item[0])
    if len(found) < 2:
        return ""
    sentences = []
    for later in range(1, len(found)):
        for earlier in range(later):
            days = abs((found[later][2] - found[earlier][2]).days)
            relation = "before" if found[later][2] < found[earlier][2] else "after"
            unit = "day" if days == 1 else "days"
            sentences.append(f"{found[later][1]} is {days} {unit} {relation} {found[earlier][1]}.")
    return " ".join(sentences)


def withhold_case(context: str) -> str | None:
    """The context with the case sentence that contains a number removed.

    Returns None when there is no `case:` section or no such sentence. The policy text stays.
    """
    marker = "case:"
    index = context.lower().find(marker)
    if index < 0:
        return None
    head = context[: index + len(marker)]
    parts = re.split(r"(?<=[.!?])\s+", context[index + len(marker) :].strip())
    for position, part in enumerate(parts):
        if re.search(r"\d", part):
            kept = " ".join(parts[:position] + parts[position + 1 :]).strip()
            if not kept:
                return None
            return f"{head} {kept}"
    return None


def extra_cases(rows: list[EvalExample]) -> list[EvalExample]:
    """Date-count copies and withheld-fact copies of rows that are policies."""
    extra: list[EvalExample] = []
    for row in rows:
        if row.uniform:
            continue
        context = row.request.context or ""
        facts = date_facts(context)
        if facts and "date_facts:" not in context:
            extra.append(
                row.model_copy(
                    update={
                        "request": row.request.model_copy(
                            update={"context": f"{context}\n\ndate_facts: {facts}"}
                        )
                    }
                )
            )
        withheld = withhold_case(context)
        if withheld is not None and withheld != context:
            extra.append(
                row.model_copy(
                    update={
                        "request": row.request.model_copy(update={"context": withheld}),
                        "uniform": True,
                    }
                )
            )
    return extra
