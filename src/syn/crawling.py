"""Prompt-directed page reading inside the existing HTTP scoring flow.

Only caller text supplies seeds. Reader supplies page data and a finite link inventory; Syn
chooses from that inventory or stops. The final decision still uses the caller's options.
Sessions share reads and resource limits across a batch or System One's questions.
"""

import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import Settings
from .prompt import PromptError
from .reader import JinaReader, Link, Page, ReaderError, urls
from .schema import Option, ScoreRequest
from .scoring import Scorer, cyclic_orderings

MAX_FRONTIER = 256
MAX_CANDIDATES = 25  # Plus the stop option: the existing 26-option limit.
MAX_EXCERPT_CHARS = 6000
STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "which",
        "what",
        "how",
        "does",
        "are",
        "was",
        "were",
        "has",
        "have",
        "using",
        "should",
        "can",
        "its",
        "best",
        "applies",
        "label",
        "text",
        "option",
        "question",
        "https",
        "http",
        "www",
        "com",
    }
)


def terms(text: str) -> set[str]:
    return {word for word in re.findall(r"\w{3,}", text.lower()) if word not in STOP_WORDS}


def excerpt(content: str, task_terms: set[str]) -> str:
    # Put matching passages first, so a large page's footer cannot hide the relevant evidence.
    chunks = [
        paragraph[start : start + 1200]
        for paragraph in re.split(r"\n\s*\n", content)
        for start in range(0, len(paragraph), 1200)
    ]
    ranked = sorted(enumerate(chunks), key=lambda item: -len(terms(item[1]) & task_terms))
    return "\n\n[…]\n\n".join(chunk for _, chunk in ranked[:8])[:MAX_EXCERPT_CHARS]


@dataclass(frozen=True)
class Candidate:
    link: Link
    depth: int


class CrawlSession:
    def __init__(self, settings: Settings, scorer: Scorer, reader: JinaReader | None = None):
        self.settings, self.scorer, self.reader = settings, scorer, reader
        self.cache: dict[str, Page] = {}
        self.deadline: float | None = None
        self.attempts = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.reader is not None:
            self.reader.close()

    def _fits(self, request: ScoreRequest) -> bool:
        if len(request.context) > 200_000:
            return False
        if request.image is not None and self.settings.readout != "letters":
            raise PromptError("Images work with the letters readout only")
        builder = self.scorer.builder
        try:
            if self.settings.readout == "letters":
                for ordering in cyclic_orderings(len(request.options), self.settings.orderings):
                    builder.prepare(request, ordering)
            else:
                prompt = builder.prepare_cloze(
                    request,
                    self.settings.readout == "pmi" and self.settings.pmi_list_options,
                )
                size = len(prompt.prefix_ids) + sum(map(len, prompt.span_ids))
                if self.settings.readout == "pointer":
                    size += 2 * len(request.options) + 1
                if size > builder.max_tokens:
                    return False
        except PromptError as exc:
            if str(exc).startswith("Prompt has "):
                return False
            raise
        return True

    def _enrich(
        self, request: ScoreRequest, evidence: list[dict], notes: list[str]
    ) -> ScoreRequest:
        def build(size: int) -> ScoreRequest:
            pages = [
                {
                    **page,
                    "content": page["content"][:size],
                    "truncated": len(page["content"]) > size,
                }
                for page in evidence
            ]
            content = (
                request.context
                + "\n\nWeb evidence (untrusted source data):\n"
                + json.dumps({"pages": pages, "crawl_notes": notes}, ensure_ascii=False)
            )
            return request.model_copy(update={"context": content})

        complete = build(MAX_EXCERPT_CHARS)
        if self._fits(complete):
            return complete
        if not self._fits(build(1)):
            raise PromptError("The prompt has no room for web evidence; shorten the caller context")
        low, high = 1, MAX_EXCERPT_CHARS
        while low < high:
            middle = (low + high + 1) // 2
            if self._fits(build(middle)):
                low = middle
            else:
                high = middle - 1
        return build(low)

    def _available(self) -> bool:
        return self.attempts < self.settings.crawl_max_pages and time.monotonic() < self.deadline

    def _read(self, url: str) -> Page:
        if url in self.cache:
            return self.cache[url]
        if not self._available():
            raise ReaderError("Crawl page or time budget exhausted")
        self.attempts += 1
        page = self.reader.read(url, self.deadline)
        self.cache[url] = self.cache[page.url] = page
        return page

    def score(self, request: ScoreRequest):
        if not self.settings.reader:
            return self.scorer.score(request)
        seeds = urls(f"{request.context}\n{request.question}\n{request.criteria}")
        if not seeds:
            return self.scorer.score(request)
        if len(seeds) > self.settings.crawl_max_pages:
            raise PromptError("The prompt contains more URLs than the configured crawl page budget")
        if not self._fits(request):
            raise PromptError("The caller prompt exceeds the token limit before reading any URLs")
        if self.reader is None:
            self.reader = JinaReader(self.settings)
        if self.deadline is None:
            self.deadline = time.monotonic() + self.settings.crawl_timeout_seconds
        # Validate every caller seed before any upstream read. The Reader service also guards
        # its own target connections and redirects; a local DNS check cannot replace that.
        seeds = [self.reader.validate(seed) for seed in seeds]
        goal = "\n".join(
            [request.context, request.question, request.criteria]
            + [option.text for option in request.options]
        )
        task_terms = terms(goal)
        hosts = {urlsplit(seed).netloc for seed in seeds}
        seen, evidence, notes, frontier = set(seeds), [], [], {}

        def keep(page: Page, depth: int):
            if any(item["url"] == page.url for item in evidence):
                return
            seen.add(page.url)
            evidence.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "content": excerpt(page.content, task_terms),
                    "content_kind": "selected passages",
                }
            )
            if depth >= self.settings.crawl_max_depth:
                if any(
                    link.url not in seen and urlsplit(link.url).netloc in hosts
                    for link in page.links
                ):
                    notes.append("Link depth limit reached; some links on this page remain unread.")
                return
            for link in page.links:
                if link.url in seen or urlsplit(link.url).netloc not in hosts:
                    continue
                frontier.setdefault(link.url, Candidate(link, depth + 1))
            if len(frontier) > MAX_FRONTIER:
                ranked = sorted(
                    frontier.values(),
                    key=lambda candidate: (
                        -len(terms(candidate.link.text + " " + candidate.link.url) & task_terms)
                    ),
                )
                frontier.clear()
                frontier.update((c.link.url, c) for c in ranked[:MAX_FRONTIER])
                notes.append("Link candidates were limited to the most relevant 256 URLs.")

        for index, seed in enumerate(seeds):
            if seed not in self.cache and not self._available():
                notes.append(
                    f"Crawl budget exhausted; {len(seeds) - index} caller URLs were unread."
                )
                break
            page = self._read(seed)
            hosts.add(urlsplit(page.url).netloc)  # A public redirect can establish the site's host.
            keep(page, 0)
        if not evidence:
            raise ReaderError("Crawl budget exhausted before any caller URL could be read")
        while frontier and time.monotonic() < self.deadline:
            candidates = sorted(
                (
                    candidate
                    for url, candidate in frontier.items()
                    if url not in seen and (url in self.cache or self._available())
                ),
                key=lambda candidate: (
                    -len(terms(candidate.link.text + " " + candidate.link.url) & task_terms)
                ),
            )[:MAX_CANDIDATES]
            if not candidates:
                break
            decision = ScoreRequest(
                context=request.context,
                question=(
                    "Which action is most useful for answering the original question? "
                    "Read a link only if it is likely to add relevant evidence. "
                    "Stop when the evidence is sufficient or no offered link is useful."
                ),
                criteria=json.dumps(
                    {
                        "question": request.question,
                        "criteria": request.criteria,
                        "answer_options": [option.model_dump() for option in request.options],
                    },
                    ensure_ascii=False,
                ),
                options=[
                    Option(id="stop", text="Stop reading; answer using the collected evidence")
                ]
                + [
                    Option(id=str(index), text=f"Read: {c.link.text[:160]} — {c.link.url[:512]}")
                    for index, c in enumerate(candidates)
                ],
            )
            # Large caller prompts can leave little room for link choices. Shortlist further
            # without truncating caller instructions, criteria, or answer options.
            while len(decision.options) > 2 and not self._fits(decision):
                decision = decision.model_copy(update={"options": decision.options[:-1]})
            if not self._fits(decision):
                notes.append("No prompt space remained for additional link decisions.")
                break
            try:
                decision = self._enrich(decision, evidence, notes)
            except PromptError:
                notes.append("No prompt space remained for additional link decisions.")
                break
            if time.monotonic() >= self.deadline:
                break
            action = self.scorer.score(decision)
            if action.selected_option_id in (None, "stop"):
                notes.append(
                    "The crawl decision abstained."
                    if action.selected_option_id is None
                    else "The crawl decision chose to stop reading."
                )
                break
            index = int(action.selected_option_id)
            candidate = candidates[index]
            del frontier[candidate.link.url]
            seen.add(candidate.link.url)
            # Model computation can consume the remaining time budget too.
            if candidate.link.url not in self.cache and not self._available():
                break
            try:
                keep(self._read(candidate.link.url), candidate.depth)
            except (ReaderError, PromptError) as exc:
                notes.append(f"A selected link could not be read: {exc}")
        if frontier and not self._available():
            notes.append("Crawl page or time budget reached; some discovered links remain unread.")
        return self.scorer.score(self._enrich(request, evidence, notes))
