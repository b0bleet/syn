import asyncio
import json
import math

import pytest
from test_shorthand import build_client
from test_systemone import TICKET, build_app

from syn.api import create_app
from syn.http_job import serve


def replay(app, **request):
    return asyncio.run(serve(app, request))


@pytest.fixture
def app():
    # The shorthand fake scores "spam" above "legit" by label text.
    return build_client().app


def test_get_shorthand_keeps_escaped_commas_and_the_query(app):
    out = replay(app, method="GET", path="/spam,legit/Win+a+free+iPhone")
    assert out == {
        "status": 200,
        "headers": out["headers"],
        "body": "spam\n",
    }
    assert out["headers"]["content-type"].startswith("text/plain")
    assert out["headers"]["x-syn-selected"] == "spam"
    verbose = replay(app, method="GET", path="/spam,legit/hi?verbose=1")
    assert json.loads(verbose["body"])["results"][0]["label"] == "spam"
    # %2C stays inside one label: three labels here, not four.
    scores = {"spam": math.log(0.6), "legit": math.log(0.2), "a,b": math.log(0.2)}
    three = replay(
        build_client(scores=scores).app, method="GET", path="/spam,legit,a%2Cb/hi?format=json"
    )
    assert [s["id"] for s in json.loads(three["body"])["scores"]] == ["spam", "legit", "a,b"]


def test_post_bodies_and_errors_pass_through(app):
    ok = replay(
        app,
        method="POST",
        path="/",
        headers={"Content-Type": "application/json", "Cookie": "dropped"},
        body=json.dumps({"input": "hi", "labels": ["spam", "legit"]}),
    )
    assert ok["status"] == 200 and json.loads(ok["body"])["results"][0]["label"] == "spam"
    bad = replay(app, method="GET", path="/spam/hi")
    assert bad["status"] == 422 and "2 to 26 labels" in bad["body"]
    assert replay(app, method="DELETE", path="/v1/score")["status"] == 405


def test_systemone_through_the_job():
    out = replay(
        build_app(),
        method="POST",
        path="/v1/systemone",
        headers={"content-type": "application/json"},
        body=json.dumps(TICKET),
    )
    assert out["status"] == 200
    assert out["headers"]["x-typesafe-request-id"]
    assert json.loads(out["body"])["answers"]["tone"]["choice"] == "frustrated"


def test_replay_needs_no_lifespan_and_rejects_bad_paths():
    app = create_app(scorer=build_client().app.state.scorer)
    assert replay(app, method="GET", path="/health")["status"] == 200
    for path in [None, "health", "https://elsewhere/x"]:
        with pytest.raises(ValueError, match="absolute path"):
            replay(app, method="GET", path=path)
