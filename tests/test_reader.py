import json
import time

import httpx
import pytest

from syn import reader
from syn.config import Settings
from syn.prompt import PromptError
from syn.reader import JinaReader, ReaderError, normalize_url, urls


def public_resolver(host, port):
    return ["93.184.216.34"]


def test_prompt_urls_preserve_queries_and_spa_routes_and_deduplicate_anchors():
    assert urls(
        "Read (https://EXAMPLE.com/a?q=hello%20world&n=2), then "
        "[the same page](https://example.com/a?q=hello%20world&n=2#section). "
        "Also https://example.com/#/settings and https://example.com/wiki/A_(B)."
    ) == [
        "https://example.com/a?q=hello%20world&n=2",
        "https://example.com/#/settings",
        "https://example.com/wiki/A_(B)",
    ]
    assert normalize_url("http://EXAMPLE.com:80") == "http://example.com/"
    assert urls("Read `https://example.com` or “https://example.com./”.") == [
        "https://example.com/"
    ]
    assert normalize_url("https://[2606:4700::1111]:443/#section") == ("https://[2606:4700::1111]/")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https:///missing",
        "https://user:secret@example.com/",
        "https://example.com:0/",
        "https://example.com:99999/",
        "https://example.com\\path",
        "https://example.com/\nprivate",
        "https://example.com/" + "x" * 4096,
    ],
)
def test_invalid_url_syntax_is_refused(url):
    with pytest.raises(PromptError, match="crawl URL"):
        normalize_url(url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/", "http://169.254.169.254/", "https://[::ffff:127.0.0.1]/"]
)
def test_literal_private_addresses_are_refused_without_reader_or_dns_calls(url):
    def never(*args):
        pytest.fail("a private target must not be fetched or resolved")

    with httpx.Client(transport=httpx.MockTransport(never)) as client:
        service = JinaReader(Settings(), client, resolver=never)
        with pytest.raises(PromptError, match="public address"):
            service.read(url, time.monotonic() + 30)


def test_mixed_public_and_private_dns_answers_are_refused():
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("no read"))
    ) as client:
        service = JinaReader(Settings(), client, lambda *_: ["93.184.216.34", "10.0.0.1"])
        with pytest.raises(PromptError, match="public address"):
            service.validate("https://example.com")


@pytest.mark.parametrize("as_pairs", [False, True])
def test_reader_request_and_page_link_inventory(as_pairs):
    observed = []

    def serve(request):
        observed.append(request)
        pairs = [
            ["Pricing", "/pricing#plans"],
            ["Pricing again", "https://example.com/pricing"],
            ["Not a page", "javascript:alert(1)"],
            ["Credentials", "https://a:b@example.com/"],
        ]
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": {
                    "url": "https://example.com/",
                    "title": "Example",
                    "content": "Public content",
                    "links": pairs if as_pairs else dict(pairs),
                    "httpStatus": 200,
                },
            },
        )

    settings = Settings(reader_url="http://reader.internal:8081", jina_api_key="jina-secret")
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        page = JinaReader(settings, client, public_resolver).read(
            "https://example.com/#/home", time.monotonic() + 30
        )
    [request] = observed
    assert request.method == "POST" and str(request.url) == "http://reader.internal:8081/"
    assert json.loads(request.content) == {"url": "https://example.com/#/home"}
    assert request.headers["authorization"] == "Bearer jina-secret"
    assert request.headers["x-with-links-summary"] == "all"
    assert request.headers["x-respond-with"] == "markdown"
    assert request.headers["x-robots-txt"] == "sifty-crawl"
    assert [(link.text, link.url) for link in page.links] == [
        ("Pricing", "https://example.com/pricing")
    ]
    assert page.content == "Public content"


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(429),
        httpx.Response(302, headers={"location": "http://elsewhere/"}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"code": 500, "data": {}}),
        httpx.Response(200, json={"data": {"content": []}}),
        httpx.Response(200, json={"data": {"content": "Forbidden", "httpStatus": 403}}),
        httpx.Response(200, json={"data": {"content": ""}}),
    ],
)
def test_bad_upstream_responses_are_errors(reply):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda request: reply)) as client,
        pytest.raises(ReaderError),
    ):
        JinaReader(Settings(), client, public_resolver).read(
            "https://example.com/", time.monotonic() + 30
        )


def test_oversize_response_and_time_budget_are_bounded(monkeypatch):
    monkeypatch.setattr(reader, "MAX_BYTES", 10)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 11))
    ) as client:
        service = JinaReader(Settings(), client, public_resolver)
        with pytest.raises(ReaderError, match="too large"):
            service.read("https://example.com/", time.monotonic() + 30)
        with pytest.raises(ReaderError, match="time budget"):
            service.read("https://example.com/", time.monotonic() - 1)


def test_upstream_error_does_not_echo_secrets():
    def fail(request):
        raise httpx.ConnectError("secret-url-and-api-key", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        service = JinaReader(Settings(jina_api_key="secret-api-key"), client, public_resolver)
        with pytest.raises(ReaderError, match="ConnectError") as error:
            service.read("https://example.com/", time.monotonic() + 30)
        assert "secret" not in str(error.value)
