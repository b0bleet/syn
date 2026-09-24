"""TypeSafe System One protocol, so `typesafe-sdk` clients work against this service.

    POST /v1/systemone  {"state", "model", "questions": {name: noul | choice | score}}
    GET  /v1/models

Each question becomes one option-scoring request over the same state:

    noul    two options, yes and no; the answer is P(yes)
    choice  one option per criteria key; the answer is the most probable key
    score   one option per rubric level; the answer is the probability-weighted level

Any model name is accepted and answered by the configured checkpoint, which the response names.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field

from .schema import ImageReference, ScoreRequest, ScoreResponse, StrictModel

JSONContent = str | dict[str, Any] | list[Any]
MODEL_ALIAS = "syn-latest"
RELEASE_DATE = "2026-09-21"
DEFAULT_INSTRUCTIONS = {
    "noul": "Is this true?",
    "choice": "Which option best applies?",
    "score": "Which level best applies?",
}


class NoulCriteria(StrictModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[Annotated[str, Field(min_length=1, max_length=128)], JSONContent | None] = Field(
        min_length=1, max_length=26
    )


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=26)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1, max_length=32)
    # A sifty extension: an image every question is asked about, alongside the state.
    image: ImageReference | None = None


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0, le=1)


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float = Field(ge=0, le=1)
    legend: dict[str, JSONContent]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int
    # Nothing is generated; answers are read from option probabilities.
    output_tokens: int = 0


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelInfo(StrictModel):
    name: str
    description: str
    release_date: str


class ModelList(StrictModel):
    models: list[ModelInfo]


def render(content: JSONContent, indent: int = 0) -> str:
    """Flatten a string, object, or array into the text the model sees.

    Field names stay as labels, one `key: value` line each. Nested objects and arrays are
    indented under their key. A string is kept as written.
    """
    pad = "  " * indent
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bool):
        return "true" if content else "false"
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in content)
    if isinstance(content, dict):
        lines = []
        for key, value in content.items():
            if isinstance(value, (dict, list)):
                rendered = render(value, indent + 1)
                lines.append(f"{pad}{key}:\n{rendered}" if rendered else f"{pad}{key}:")
            else:
                lines.append(f"{pad}{key}: {render(value)}")
        return "\n".join(lines)
    return str(content)


def described(name: str, description: JSONContent | None) -> str:
    return name if description is None else f"{name}: {render(description)}"


def to_request(state: str, question: Question, image: str | None = None) -> ScoreRequest | None:
    """The scoring request for one question; None when a single option answers it outright."""
    if isinstance(question, NoulQuestion):
        criteria = question.criteria or NoulCriteria()
        options = [
            {"id": "yes", "text": described("Yes", criteria.true)},
            {"id": "no", "text": described("No", criteria.false)},
        ]
    elif isinstance(question, ChoiceQuestion):
        options = [{"id": k, "text": described(k, v)} for k, v in question.criteria.items()]
    else:
        options = [{"id": str(i), "text": render(c)} for i, c in enumerate(question.criteria)]
    if len(options) == 1:
        return None
    instructions = question.instructions
    ask = DEFAULT_INSTRUCTIONS[question.type] if instructions is None else render(instructions)
    payload = {"context": state, "question": ask, "options": options}
    if image is not None:
        payload["image"] = image
    return ScoreRequest.model_validate(payload)


def to_answer(question: Question, response: ScoreResponse | None):
    probabilities = {s.id: s.probability for s in response.scores} if response else None
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=probabilities["yes"])
    if isinstance(question, ChoiceQuestion):
        if response is None:
            [only] = question.criteria
            return ChoiceAnswer(choice=only, confidence=1.0, probabilities={only: 1.0})
        return ChoiceAnswer(
            choice=response.best_option_id,
            confidence=response.confidence,
            probabilities=probabilities,
        )
    legend = {str(i): c for i, c in enumerate(question.criteria)}
    if response is None:
        return ScoreAnswer(score=0.0, confidence=1.0, legend=legend, probabilities={"0": 1.0})
    return ScoreAnswer(
        score=sum(int(level) * p for level, p in probabilities.items()),
        confidence=response.confidence,
        legend=legend,
        probabilities=probabilities,
    )


def system_one(
    score: Callable[[ScoreRequest], ScoreResponse], request: SystemOneRequest, model: str
) -> SystemOneResponse:
    """Answer every question about the state. Raises ValueError for an unscorable question."""
    state = render(request.state)
    # Build every request first, so an invalid question fails before any scoring runs.
    built = {name: to_request(state, q, request.image) for name, q in request.questions.items()}
    answers, tokens = {}, 0
    for name, question in request.questions.items():
        response = score(built[name]) if built[name] is not None else None
        tokens += response.prompt_tokens if response else 0
        answers[name] = to_answer(question, response)
    return SystemOneResponse(model=model, answers=answers, usage=Usage(input_tokens=tokens))


def models(model: str, readout: str) -> ModelList:
    return ModelList(
        models=[
            ModelInfo(
                name=MODEL_ALIAS,
                description=f"Zero-shot option scoring on {model} ({readout} readout).",
                release_date=RELEASE_DATE,
            )
        ]
    )
