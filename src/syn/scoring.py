import json
import math
import threading
import time
import uuid
from datetime import UTC, datetime

from .backends import Backend, BackendError
from .config import Settings
from .prompt import CLOZE_VERSION, HEAD_VERSION, POINTER_VERSION, PromptBuilder
from .schema import OptionScore, ScoreRequest, ScoreResponse

# log_softmax can exceed zero by rounding; anything larger is a backend bug, not noise.
LOG_PROB_TOLERANCE = 1e-6


def probabilities(log_scores: list[float], temperature: float = 1.0) -> list[float]:
    if not log_scores or not all(math.isfinite(x) for x in log_scores):
        raise ValueError("Scores must be nonempty and finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    peak = max(log_scores)
    weights = [math.exp((x - peak) / temperature) for x in log_scores]
    total = sum(weights)
    return [x / total for x in weights]


def cyclic_orderings(count: int, limit: int) -> list[list[int]]:
    """Option index at each position, for each ordering to score.

    With limit 0 (or limit >= count) there are `count` orderings and every option takes every
    position exactly once, which cancels position bias in the averaged scores.
    """
    shifts = count if limit == 0 else min(limit, count)
    return [[(position + shift) % count for position in range(count)] for shift in range(shifts)]


class Scorer:
    def __init__(
        self,
        settings: Settings,
        builder: PromptBuilder,
        backend: Backend,
        revision_commit: str | None = None,
        head=None,
        head_sha256: str | None = None,
        head_temperature: float | None = None,
        pointer=None,
    ):
        self.settings, self.builder, self.backend = settings, builder, backend
        self.revision_commit = revision_commit
        self.head, self.head_sha256 = head, head_sha256
        # The temperature fitted when the head was trained. SYN_TEMPERATURE, when set, wins.
        self.head_temperature = head_temperature
        # A loaded PointerReadout: its head, temperature, delimiters, and checksum.
        self.pointer = pointer
        if pointer is not None:
            self.head_sha256, self.head_temperature = pointer.sha256, pointer.temperature
        if settings.readout == "head" and head is None:
            raise ValueError("readout=head needs a loaded AttentionHead")
        if settings.readout == "pointer" and pointer is None:
            raise ValueError("readout=pointer needs a loaded PointerReadout")
        self.log_lock = threading.Lock()
        if settings.log_path:
            settings.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def prompt_version(self) -> str:
        if self.settings.readout == "pmi":
            return CLOZE_VERSION
        if self.settings.readout == "head":
            return HEAD_VERSION
        if self.settings.readout == "pointer":
            return POINTER_VERSION
        return self.builder.version

    def score(self, request: ScoreRequest) -> ScoreResponse:
        started = time.perf_counter()
        if self.settings.readout == "pmi":
            return self._score_pmi(request, started)
        if self.settings.readout == "head":
            return self._score_head(request, started)
        if self.settings.readout == "pointer":
            return self._score_pointer(request, started)
        return self._score_letters(request, started)

    def close(self):
        self.backend.close()

    def _score_letters(self, request: ScoreRequest, started: float) -> ScoreResponse:
        count = len(request.options)
        orderings = cyclic_orderings(count, self.settings.orderings)
        prompts = [self.builder.prepare(request, order) for order in orderings]
        result = self.backend.score(prompts)
        rows = result.log_probs
        if len(rows) != len(prompts) or any(len(row) != count for row in rows):
            raise BackendError("Backend returned the wrong number of scores")
        totals = [0.0] * count
        wins = [0] * count
        masses = []
        for order, row in zip(orderings, rows, strict=True):
            if any(not math.isfinite(x) or x > LOG_PROB_TOLERANCE for x in row):
                raise BackendError("Backend returned invalid log probabilities")
            row = [min(x, 0.0) for x in row]
            mass = sum(math.exp(x) for x in row)
            if mass > 1.0001:
                raise BackendError("Backend returned invalid probability mass")
            masses.append(min(1.0, mass))
            for position, option in enumerate(order):
                totals[option] += row[position]
            wins[order[max(range(count), key=row.__getitem__)]] += 1
        return self._respond(
            request,
            scores=[total / len(orderings) for total in totals],
            wins=wins,
            orderings=len(orderings),
            mass=sum(masses) / len(masses),
            prompt_tokens=max(len(p.input_ids) for p in prompts),
            rewritten=prompts[0].rewritten_control_tokens,
            started=started,
            queue_ms=result.wait_ms,
            backend_ms=result.compute_ms,
        )

    def _score_pmi(self, request: ScoreRequest, started: float) -> ScoreResponse:
        """Each option's text likelihood given the context, minus its likelihood without it.

        All spans are scored in one forward under a block mask, so no option sees another and no
        letters or positions are involved. The prior subtraction removes the model's preference
        for common label wordings, which is what made "not spam" win on frequency alone.
        """
        cloze = self.builder.prepare_cloze(request, self.settings.pmi_list_options)
        count = len(request.options)
        conditional = self.backend.score_spans(cloze.prefix_ids, cloze.span_ids)
        prior = self.backend.score_spans(cloze.prior_prefix_ids, cloze.span_ids)
        rows = []
        for result in (conditional, prior):
            if len(result.log_probs) != 1 or len(result.log_probs[0]) != count:
                raise BackendError("Backend returned the wrong number of span scores")
            row = result.log_probs[0]
            if any(not math.isfinite(x) or x > LOG_PROB_TOLERANCE for x in row):
                raise BackendError("Backend returned invalid span log probabilities")
            rows.append(row)
        scores = [c - p for c, p in zip(rows[0], rows[1], strict=True)]
        best = max(range(count), key=lambda i: scores[i])
        queue = None
        if conditional.wait_ms is not None and prior.wait_ms is not None:
            queue = conditional.wait_ms + prior.wait_ms
        return self._respond(
            request,
            scores=scores,
            wins=[int(i == best) for i in range(count)],
            orderings=1,
            mass=None,
            prompt_tokens=len(cloze.prefix_ids) + sum(len(span) for span in cloze.span_ids),
            rewritten=cloze.rewritten_control_tokens,
            started=started,
            queue_ms=queue,
            backend_ms=conditional.compute_ms + prior.compute_ms,
        )

    def _score_head(self, request: ScoreRequest, started: float) -> ScoreResponse:
        """Frozen-backbone features for context and standalone options, scored by the head.

        The rendering is the same one the features were cached with, so a head trained offline
        sees exactly what it saw in training. Its softmax was fitted with cross-entropy on labels,
        which is what makes these probabilities mean something.
        """
        import numpy as np
        import torch

        cloze = self.builder.prepare_cloze(request, list_options=False)
        count = len(request.options)
        result = self.backend.features(cloze.prefix_ids, cloze.span_ids)
        context = torch.from_numpy(np.asarray(result.context, dtype=np.float32))[None]
        options = torch.from_numpy(np.asarray(result.options, dtype=np.float32))[None]
        if options.shape[1] != count:
            raise BackendError("Backend returned the wrong number of option vectors")
        with torch.inference_mode():
            logits = self.head(
                context,
                torch.ones(1, context.shape[1], dtype=torch.bool),
                options,
                torch.ones(1, count, dtype=torch.bool),
            )[0].tolist()
        if any(not math.isfinite(x) for x in logits):
            raise BackendError("Head returned non-finite logits")
        best = max(range(count), key=lambda i: logits[i])
        return self._respond(
            request,
            scores=logits,
            wins=[int(i == best) for i in range(count)],
            orderings=1,
            mass=None,
            prompt_tokens=len(cloze.prefix_ids) + sum(len(span) for span in cloze.span_ids),
            rewritten=cloze.rewritten_control_tokens,
            started=started,
            queue_ms=result.wait_ms,
            backend_ms=result.compute_ms,
            temperature=self.head_temperature,
        )

    def _score_pointer(self, request: ScoreRequest, started: float) -> ScoreResponse:
        """The adapted backbone's decide and option states, scored by the pointer head.

        Every option span sees only the prefix and itself, and the decide token sees all of
        them from a fixed position, so the logits are invariant to option order by construction.
        """
        import numpy as np
        import torch

        cloze = self.builder.prepare_cloze(request, list_options=False)
        count = len(request.options)
        result = self.backend.pointer_states(
            cloze.prefix_ids, cloze.span_ids, self.pointer.delimiters
        )
        decide = torch.from_numpy(np.asarray(result.decide, dtype=np.float32))[None]
        options = torch.from_numpy(np.asarray(result.options, dtype=np.float32))[None]
        if options.shape[1] != count:
            raise BackendError("Backend returned the wrong number of option states")
        with torch.inference_mode():
            logits = self.pointer.head(decide, options)[0].tolist()
        if any(not math.isfinite(x) for x in logits):
            raise BackendError("Pointer head returned non-finite logits")
        best = max(range(count), key=lambda i: logits[i])
        return self._respond(
            request,
            scores=logits,
            wins=[int(i == best) for i in range(count)],
            orderings=1,
            mass=None,
            prompt_tokens=len(cloze.prefix_ids) + sum(len(span) + 2 for span in cloze.span_ids) + 1,
            rewritten=cloze.rewritten_control_tokens,
            started=started,
            queue_ms=result.wait_ms,
            backend_ms=result.compute_ms,
            temperature=self.head_temperature,
        )

    def _respond(
        self,
        request: ScoreRequest,
        *,
        scores: list[float],
        wins: list[int],
        orderings: int,
        mass: float | None,
        prompt_tokens: int,
        rewritten: int,
        started: float,
        queue_ms: float | None,
        backend_ms: float,
        temperature: float | None = None,
    ) -> ScoreResponse:
        count = len(request.options)
        if temperature is None or "temperature" in self.settings.model_fields_set:
            temperature = self.settings.temperature
        probs = probabilities(scores, temperature)
        best = max(range(count), key=probs.__getitem__)
        chance = 1.0 / count
        confidence = min(1.0, max(0.0, (probs[best] - chance) / (1.0 - chance)))
        agreement = wins[best] / orderings
        reasons = []
        if probs[best] < self.settings.abstain_threshold:
            reasons.append("low_probability")
        if confidence < self.settings.min_confidence:
            reasons.append("low_confidence")
        if agreement < self.settings.min_ordering_agreement:
            reasons.append("ordering_disagreement")
        response = ScoreResponse(
            request_id=str(uuid.uuid4()),
            model=self.settings.model,
            revision=self.settings.revision,
            revision_commit=self.revision_commit,
            backend=self.settings.backend,
            prompt_version=self.prompt_version,
            readout=self.settings.readout,
            head_sha256=self.head_sha256,
            scores=[
                OptionScore(id=option.id, log_probability=s, probability=p, wins=w)
                for option, s, p, w in zip(request.options, scores, probs, wins, strict=True)
            ],
            best_option_id=request.options[best].id,
            selected_option_id=None if reasons else request.options[best].id,
            abstained=bool(reasons),
            abstain_reasons=reasons,
            confidence=confidence,
            ordering_agreement=agreement,
            orderings_scored=orderings,
            label_probability_mass=mass,
            temperature=temperature,
            abstain_threshold=self.settings.abstain_threshold,
            min_confidence=self.settings.min_confidence,
            min_ordering_agreement=self.settings.min_ordering_agreement,
            prompt_tokens=prompt_tokens,
            rewritten_control_tokens=rewritten,
            latency_ms=(time.perf_counter() - started) * 1000,
            queue_ms=queue_ms,
            backend_ms=backend_ms,
        )
        if self.settings.log_path:
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "request": request.model_dump(),
                "response": response.model_dump(),
            }
            with self.log_lock, self.settings.log_path.open("a") as out:
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return response
