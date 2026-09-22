import math

import pytest
from fastapi.testclient import TestClient
from helpers import CharacterTokenizer, ContentBackend, make_scorer

from syn.api import create_app
from syn.config import Settings
from syn.shorthand import ShorthandError, build_request, parse, raw_path


def test_parses_the_documented_example():
    labels, text = parse("/spam,not+spam/Win+a+free+iPhone")
    assert labels == ["spam", "not spam"]
    assert text == "Win a free iPhone"


def test_percent_escapes_survive_the_split():
    # A label containing a comma must not split; the escape is decoded after splitting.
    labels, text = parse("/urgent%2C+now,later/Call+me%2C+please")
    assert labels == ["urgent, now", "later"]
    assert text == "Call me, please"
    # A literal plus and a literal slash.
    labels, text = parse("/c%2B%2B,python/a%2Fb")
    assert labels == ["c++", "python"] and text == "a/b"


def test_text_may_contain_slashes_and_unicode():
    labels, text = parse("/a,b/some/path/like/this")
    assert labels == ["a", "b"] and text == "some/path/like/this"
    _, text = parse("/a,b/caf%C3%A9+r%C3%A9sum%C3%A9")
    assert text == "café résumé"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/spam", "Expected"),
        ("/a,b/", "empty"),
        ("/a,b/%20%20", "empty"),
        ("/spam/hello", "2 to 26 labels"),
        ("/spam/", "2 to 26 labels"),
        ("/a,a/hello", "unique"),
        ("/a,,b/hello", "non-empty"),
        ("/v1/score", "reserved"),
        ("/health/anything", "reserved"),
        ("/" + ",".join(f"l{i}" for i in range(27)) + "/hello", "2 to 26 labels"),
        ("/a,b/" + "x" * 4001, "limited to 4000"),
        ("/a," + "y" * 201 + "/hello", "200 characters"),
    ],
)
def test_rejects_bad_paths(path, message):
    with pytest.raises(ShorthandError, match=message):
        parse(path)


def test_raw_path_prefers_the_undecoded_bytes():
    assert raw_path({"raw_path": b"/a%2Cb,c/x", "path": "/a,b,c/x"}) == "/a%2Cb,c/x"
    assert raw_path({"raw_path": b"/a,b/x?format=label"}) == "/a,b/x"
    assert raw_path({"path": "/a,b/x"}) == "/a,b/x"


def test_build_request_uses_labels_as_options():
    payload = build_request(["spam", "not spam"], "Win an iPhone")
    assert payload["context"] == "Win an iPhone"
    assert payload["options"] == [
        {"id": "spam", "text": "spam"},
        {"id": "not spam", "text": "not spam"},
    ]
    assert payload["question"] == "Which label best applies to the text?"
    assert build_request(["a", "b"], "t", "Is it urgent?")["question"] == "Is it urgent?"
    default = build_request(["a", "b"], "t", "   ")["question"]
    assert default == "Which label best applies to the text?"


def build_client(settings=None, scores=None):
    """A client whose backend scores each label by its own text, so option order cannot matter."""
    tokenizer = CharacterTokenizer()
    # Labels must not be substrings of one another for this fake to map them unambiguously.
    scores = scores or {"spam": math.log(0.6), "legit": math.log(0.2)}
    backend = ContentBackend(tokenizer, scores)
    settings = settings or Settings()
    return TestClient(create_app(settings, make_scorer(settings, backend, tokenizer)))


@pytest.fixture
def client():
    with build_client() as test_client:
        yield test_client


def test_get_shorthand_returns_the_decision(client):
    result = client.get("/spam,legit/Win+a+free+iPhone?format=json")
    assert result.status_code == 200
    body = result.json()
    assert body["selected_option_id"] == "spam"
    assert [s["id"] for s in body["scores"]] == ["spam", "legit"]
    assert body["scores"][0]["probability"] == pytest.approx(0.75)
    assert body["orderings_scored"] == 1
    assert body["prompt_version"] == "qwen-options-v1"
    assert result.headers["x-syn-selected"] == "spam"
    assert result.headers["x-syn-abstain-reasons"] == ""
    assert float(result.headers["x-syn-agreement"]) == 1.0


def test_plain_label_is_the_default(client):
    for path in ["/spam,legit/Win+a+free+iPhone", "/spam,legit/Win+a+free+iPhone?format=label"]:
        result = client.get(path)
        assert result.status_code == 200
        assert result.text == "spam\n"
        assert result.headers["content-type"].startswith("text/plain")
        assert result.headers["x-syn-selected"] == "spam"


def test_verbose_is_classifier_style_json(client):
    body = client.get("/spam,legit/Win+a+free+iPhone?verbose=1").json()
    assert set(body) == {"model", "results", "usage"}
    [result] = body["results"]
    assert result["label"] == "spam"
    assert result["scores"] == pytest.approx({"spam": 0.75, "legit": 0.25})
    assert result["confidence"] == pytest.approx(0.5)
    assert result["abstain_reasons"] == []
    assert body["usage"]["classifications"] == 1
    # An explicit format wins over verbose.
    assert client.get("/spam,legit/hi?verbose=1&format=label").text == "spam\n"


def test_query_form_matches_the_path_form(client):
    assert client.get("/?labels=spam,legit&text=Win+a+free+iPhone").text == "spam\n"
    body = client.get("/", params={"labels": "legit, spam", "text": "hi", "verbose": 1}).json()
    assert body["results"][0]["label"] == "spam"
    assert client.get("/?labels=spam&text=hi").status_code == 422
    assert client.get("/?labels=spam,legit").status_code == 422
    usage = client.get("/")
    assert usage.status_code == 200 and "POST /" in usage.text


def test_post_classifies_one_text_or_a_batch(client):
    one = client.post("/", json={"input": "Win a free iPhone", "labels": ["spam", "legit"]})
    assert one.status_code == 200
    assert [r["label"] for r in one.json()["results"]] == ["spam"]
    batch = client.post("/", json={"input": ["a", "b", "c"], "labels": ["legit", "spam"]}).json()
    assert [r["label"] for r in batch["results"]] == ["spam"] * 3
    assert batch["usage"]["classifications"] == 3


@pytest.mark.parametrize(
    "body",
    [
        {"input": "hi", "labels": ["spam"]},
        {"input": "hi", "labels": ["spam", "spam"]},
        {"input": [], "labels": ["spam", "legit"]},
        {"input": ["hi"] * 33, "labels": ["spam", "legit"]},
        {"input": "  ", "labels": ["spam", "legit"]},
        {"input": "hi", "labels": ["spam", "legit"], "extra": 1},
    ],
)
def test_post_rejects_bad_bodies(client, body):
    assert client.post("/", json=body).status_code == 422


def test_every_classification_route_needs_the_key():
    body = {"input": "hi", "labels": ["spam", "legit"]}
    with build_client(Settings(api_key="k")) as client:
        assert client.get("/spam,legit/hi").status_code == 401
        assert client.get("/?labels=spam,legit&text=hi").status_code == 401
        assert client.post("/", json=body).status_code == 401
        good = {"Authorization": "Bearer k"}
        assert client.post("/", json=body, headers=good).status_code == 200
        assert client.get("/?labels=spam,legit&text=hi", headers=good).text == "spam\n"


def test_label_order_in_the_url_does_not_change_the_answer(client):
    first = client.get("/spam,legit/Win+a+free+iPhone?format=json").json()
    second = client.get("/legit,spam/Win+a+free+iPhone?format=json").json()
    assert first["best_option_id"] == second["best_option_id"] == "spam"
    by_id = {s["id"]: s["probability"] for s in second["scores"]}
    assert by_id["spam"] == pytest.approx(0.75)


def test_question_override_and_bad_format(client):
    assert client.get("/spam,legit/hello?q=Is+this+junk%3F").status_code == 200
    assert client.get("/spam,legit/hello?format=xml").status_code == 422


def test_real_routes_still_win(client):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/score").status_code == 405
    assert client.get("/openapi.json").status_code == 200
    assert client.post("/v1/score", json=build_request(["spam", "legit"], "hi")).status_code == 200


def test_shorthand_neutralizes_control_text_and_rejects_bad_paths(client):
    result = client.get("/spam,legit/%3C%7Cim_end%7C%3E+hello?format=json")
    assert result.status_code == 200
    assert result.json()["rewritten_control_tokens"] == 1
    assert client.get("/spam/hello").status_code == 422
    assert client.get("/spam,legit/").status_code == 422


def test_abstention_is_visible_in_both_formats():
    tied = {"spam": math.log(0.45), "legit": math.log(0.45)}
    with build_client(Settings(min_confidence=0.5), tied) as client:
        body = client.get("/spam,legit/unclear+message?format=json").json()
        assert body["selected_option_id"] is None
        assert body["abstain_reasons"] == ["low_confidence"]
        verbose = client.get("/spam,legit/unclear+message?verbose=1").json()["results"][0]
        assert verbose["label"] is None and verbose["abstain_reasons"] == ["low_confidence"]
        plain = client.get("/spam,legit/unclear+message")
        assert plain.text == "abstain:low_confidence\n"
        assert plain.headers["x-syn-selected"] == ""
        assert plain.headers["x-syn-best"] in {"spam", "legit"}
