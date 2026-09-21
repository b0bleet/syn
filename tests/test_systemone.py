import asyncio
import json
import math

import httpx2
import pytest
from fastapi.testclient import TestClient
from helpers import CharacterTokenizer, make_scorer
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score

from syn.api import create_app
from syn.backends import BackendResult
from syn.config import Settings
from syn.systemone import SystemOneRequest, render, to_request

# Option descriptions the fake backend recognizes, with the probability mass it gives each.
WEIGHTS = {
    "Yes": 0.8,
    "No": 0.2,
    "Yes: A refund or charge problem": 0.9,
    "No: Anything else": 0.1,
    "calm": 0.1,
    "frustrated": 0.7,
    "angry": 0.2,
    "can wait": 0.1,
    "this week": 0.3,
    "today": 0.6,
}


class DescriptionBackend:
    """Scores each option by its description, wherever the ordering places it."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def score(self, prompts):
        rows = []
        for prompt in prompts:
            text = self.tokenizer.decode(prompt.input_ids)
            found = {
                text.index(marker): weight
                for description, weight in WEIGHTS.items()
                if (marker := f'"description": {json.dumps(description)}') in text
            }
            rows.append([math.log(found[position]) for position in sorted(found)])
        return BackendResult(rows, compute_ms=1.0, wait_ms=0.0)

    def close(self):
        pass


def build_app(settings=None):
    settings = settings or Settings(model="Qwen/Qwen3-8B")
    tokenizer = CharacterTokenizer()
    return create_app(settings, make_scorer(settings, DescriptionBackend(tokenizer), tokenizer))


@pytest.fixture
def client():
    with TestClient(build_app()) as test_client:
        yield test_client


TICKET = {
    "state": "I was charged twice. Please fix this ASAP.",
    "model": "any-model-name",
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the customer's tone?",
            "criteria": {"calm": None, "frustrated": None, "angry": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["can wait", "this week", "today"],
        },
    },
}


def test_questions_become_option_requests():
    request = SystemOneRequest.model_validate(TICKET)
    noul = to_request("state", request.questions["billing"])
    assert [(o.id, o.text) for o in noul.options] == [("yes", "Yes"), ("no", "No")]
    assert noul.question == "Is this ticket about billing?"
    tone = to_request("state", request.questions["tone"])
    assert [o.id for o in tone.options] == ["calm", "frustrated", "angry"]
    urgency = to_request("state", request.questions["urgency"])
    assert [(o.id, o.text) for o in urgency.options] == [
        ("0", "can wait"),
        ("1", "this week"),
        ("2", "today"),
    ]


def test_descriptions_and_structured_content_are_rendered():
    request = SystemOneRequest.model_validate(
        {
            "state": {"subject": "Duplicate charge"},
            "model": "any",
            "questions": {
                "a": {"type": "noul", "criteria": {"true": "Refunds", "false": None}},
                "b": {"type": "choice", "criteria": {"billing": {"team": "finance"}, "tech": None}},
            },
        }
    )
    noul = to_request("s", request.questions["a"])
    assert [o.text for o in noul.options] == ["Yes: Refunds", "No"]
    assert noul.question == "Is this true?"
    choice = to_request("s", request.questions["b"])
    assert [o.text for o in choice.options] == ["billing: team: finance", "tech"]
    assert render({"subject": "Duplicate charge", "tags": ["a", "b"]}) == (
        "subject: Duplicate charge\ntags:\n  - a\n  - b"
    )


def test_systemone_answers_every_question(client):
    result = client.post("/v1/systemone", json=TICKET)
    assert result.status_code == 200
    assert result.headers["x-typesafe-request-id"]
    body = result.json()
    assert body["model"] == "Qwen/Qwen3-8B"
    answers = body["answers"]
    assert answers["billing"] == {"type": "noul", "noul": pytest.approx(0.8)}
    tone = answers["tone"]
    assert tone["choice"] == "frustrated"
    assert tone["probabilities"] == pytest.approx({"calm": 0.1, "frustrated": 0.7, "angry": 0.2})
    urgency = answers["urgency"]
    assert urgency["score"] == pytest.approx(0.3 + 2 * 0.6)
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert urgency["probabilities"] == pytest.approx({"0": 0.1, "1": 0.3, "2": 0.6})
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0


def test_single_option_questions_need_no_scoring(client):
    body = client.post(
        "/v1/systemone",
        json={
            "state": "hi",
            "model": "m",
            "questions": {
                "only": {"type": "choice", "criteria": {"billing": None}},
                "flat": {"type": "score", "criteria": ["fine"]},
            },
        },
    ).json()
    assert body["answers"]["only"]["choice"] == "billing"
    assert body["answers"]["flat"]["score"] == 0.0
    assert body["usage"]["input_tokens"] == 0


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"q": {"type": "maybe"}},
        {"q": {"type": "choice"}},
        {"q": {"type": "choice", "criteria": {}}},
        {"q": {"type": "score", "criteria": []}},
        {"q": {"type": "noul", "surprise": 1}},
    ],
)
def test_bad_questions_are_rejected(client, questions):
    body = {"state": "hi", "model": "m", "questions": questions}
    assert client.post("/v1/systemone", json=body).status_code == 422


def test_models_and_auth():
    with TestClient(build_app(Settings(api_key="k"))) as client:
        assert client.post("/v1/systemone", json=TICKET).status_code == 401
        assert client.get("/v1/models").status_code == 401
        listed = client.get("/v1/models", headers={"Authorization": "Bearer k"}).json()
        assert listed["models"][0]["name"] == "syn-latest"


def test_the_real_typesafe_sdk_round_trips():
    """The published typesafe-sdk client, pointed at this app, parses every answer type."""

    async def main():
        client = AsyncTypeSafeClient(
            api_key="k",
            base_url="http://syn",
            model="any-model-name",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.ASGITransport(app=build_app(Settings(api_key="k"))),
        )
        async with client:
            response = await client.system_one(
                state="I was charged twice. Please fix this ASAP.",
                questions={
                    "billing": Noul(
                        instructions="Is this ticket about billing?",
                        criteria={"true": "A refund or charge problem", "false": "Anything else"},
                    ),
                    "tone": Choice(
                        instructions="What is the customer's tone?",
                        criteria={"calm": None, "frustrated": None, "angry": None},
                    ),
                    "urgency": Score(
                        instructions="How urgent is this ticket?",
                        criteria=["can wait", "this week", "today"],
                    ),
                },
            )
            listed = await client.models.list()
        return response, listed

    response, listed = asyncio.run(main())
    assert response.nouls["billing"].noul == pytest.approx(0.9)
    assert response.choices["tone"].choice == "frustrated"
    assert response.scores["urgency"].score == pytest.approx(1.5)
    assert response.scores["urgency"].legend == {0: "can wait", 1: "this week", 2: "today"}
    assert response.request_id
    assert [m.name for m in listed.models] == ["syn-latest"]
