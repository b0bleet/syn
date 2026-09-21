from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
AbstainReason = Literal["low_probability", "low_confidence", "ordering_disagreement"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Option(StrictModel):
    id: Annotated[Text, Field(max_length=128)]
    text: Annotated[Text, Field(max_length=8192)]


class ScoreRequest(StrictModel):
    context: Annotated[Text, Field(max_length=200_000)]
    question: Annotated[Text, Field(max_length=8192)]
    criteria: Annotated[str, Field(max_length=8192)] = ""
    options: list[Option] = Field(min_length=2, max_length=26)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("Option IDs must be unique")
        return self


class OptionScore(StrictModel):
    id: str
    # readout=letters: mean over orderings of the label token's full-vocabulary log probability.
    # readout=pmi: mean per-token log probability of the option text minus its context-free prior.
    # readout=head: the trained head's logit for this option.
    log_probability: float
    probability: float = Field(ge=0, le=1)
    # Orderings in which this option's label had the highest probability.
    wins: int = Field(ge=0)


class ScoreResponse(StrictModel):
    request_id: str
    model: str
    revision: str
    revision_commit: str | None
    backend: str
    prompt_version: str
    readout: Literal["letters", "pmi", "head"] = "letters"
    # sha256 of the head checkpoint when readout=head; part of the calibration profile.
    head_sha256: str | None = None
    scores: list[OptionScore]
    best_option_id: str
    selected_option_id: str | None
    abstained: bool
    abstain_reasons: list[AbstainReason]
    # Best probability rescaled so chance is 0 and certainty is 1, comparable across option counts.
    confidence: float = Field(ge=0, le=1)
    # Fraction of scored orderings whose top label belonged to the best option.
    ordering_agreement: float = Field(ge=0, le=1)
    orderings_scored: int = Field(ge=1)
    # Vocabulary mass on the label tokens (letters readout); null for pmi, where no label exists.
    label_probability_mass: float | None
    temperature: float
    abstain_threshold: float
    min_confidence: float
    min_ordering_agreement: float
    probability_semantics: Literal["conditional_option_preference"] = (
        "conditional_option_preference"
    )
    prompt_tokens: int
    # Control-token lookalikes such as `<|im_end|>` rewritten in the request text before scoring.
    rewritten_control_tokens: int = Field(default=0, ge=0)
    latency_ms: float
    # Wait for the local model lock; null for remote backends.
    queue_ms: float | None
    backend_ms: float


class ClassifyRequest(StrictModel):
    """Body for POST /: one text or a batch, and the labels to pick from."""

    input: (
        Annotated[Text, Field(max_length=200_000)]
        | Annotated[
            list[Annotated[Text, Field(max_length=200_000)]], Field(min_length=1, max_length=32)
        ]
    )
    labels: list[Annotated[Text, Field(max_length=128)]] = Field(min_length=2, max_length=26)
    question: Annotated[str, Field(max_length=8192)] | None = None

    @model_validator(mode="after")
    def unique_labels(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Labels must be unique")
        return self


class LabelResult(StrictModel):
    # Null when the service abstained; abstain_reasons says why.
    label: str | None
    # Same meaning as ScoreResponse.confidence: best probability rescaled so chance is 0.
    confidence: float = Field(ge=0, le=1)
    # Probability per label, summing to 1.
    scores: dict[str, float]
    abstain_reasons: list[AbstainReason]
    ms: float


class ClassifyUsage(StrictModel):
    classifications: int
    ms: float


class ClassifyResponse(StrictModel):
    model: str
    results: list[LabelResult]
    usage: ClassifyUsage


class EvalExample(StrictModel):
    request: ScoreRequest
    expected_option_id: str

    @model_validator(mode="after")
    def valid_answer(self):
        if self.expected_option_id not in {o.id for o in self.request.options}:
            raise ValueError("expected_option_id is missing from options")
        return self
