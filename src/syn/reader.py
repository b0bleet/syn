"""Public URL reads through Jina Reader, for the existing scoring request flow."""

import ipaddress
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .config import Settings
from .images import public, resolve
from .prompt import PromptError

MAX_BYTES = 2 * 1024 * 1024
MAX_URL_CHARS = 4096
URL_RE = re.compile(r"https?://[^\s<>\"'`“”‘’]+", re.IGNORECASE)


class ReaderError(RuntimeError):
    """Reader could not supply usable page content; maps to an upstream error."""


@dataclass(frozen=True)
class Link:
    text: str
    url: str


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    content: str
    links: tuple[Link, ...]


def normalize_url(url: str) -> str:
    """Canonicalize public web URL syntax; DNS is checked immediately before each read."""
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
        if (
            len(url) > MAX_URL_CHARS
            or parts.scheme.lower() not in ("http", "https")
            or not host
            or parts.username is not None
            or parts.password is not None
            or "\\" in url
            or any(c.isspace() or ord(c) < 32 for c in url)
        ):
            raise ValueError
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
        if not host:
            raise ValueError
        authority = f"[{host}]" if ":" in host else host
        if port is not None and port != (443 if parts.scheme.lower() == "https" else 80):
            if port == 0:
                raise ValueError
            authority += f":{port}"
        # Ordinary fragment anchors are the same page. Preserve hash-based SPA routes.
        fragment = parts.fragment if parts.fragment.startswith(("/", "!")) else ""
        return urlunsplit(
            (parts.scheme.lower(), authority, parts.path or "/", parts.query, fragment)
        )
    except (ValueError, UnicodeError) as exc:
        raise PromptError(
            "A crawl URL must be http or https, with a host and no credentials"
        ) from exc


def urls(text: str) -> list[str]:
    """URLs in caller text only; preserves query strings and balanced URL parentheses."""
    found = {}
    for match in URL_RE.finditer(text):
        url = match.group().rstrip(".,;!?")
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            while url.endswith(closing) and url.count(closing) > url.count(opening):
                url = url[:-1]
        normalized = normalize_url(url)
        found.setdefault(normalized, None)
    return list(found)


class JinaReader:
    def __init__(self, settings: Settings, client: httpx.Client | None = None, resolver=resolve):
        self.settings, self.resolver = settings, resolver
        self.own_client = client is None
        self.client = client or httpx.Client(follow_redirects=False, trust_env=False)

    def close(self):
        if self.own_client:
            self.client.close()

    def validate(self, url: str) -> str:
        url = normalize_url(url)
        parts = urlsplit(url)
        try:
            # Literal IPs do not need DNS, including IPv4-mapped IPv6 addresses.
            ipaddress.ip_address(parts.hostname)
            addresses = [parts.hostname]
        except ValueError:
            try:
                addresses = self.resolver(
                    parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
                )
            except OSError as exc:
                raise ReaderError("Could not resolve the crawl host") from exc
        if not addresses or not all(public(address) for address in addresses):
            raise PromptError("A crawl URL must point to a public address")
        return url

    def read(self, url: str, deadline: float) -> Page:
        url = self.validate(url)
        timeout = min(self.settings.reader_timeout_seconds, deadline - time.monotonic())
        if timeout <= 0:
            raise ReaderError("Crawl time budget exhausted")
        headers = {
            "Accept": "application/json",
            "X-Respond-With": "markdown",
            "X-With-Links-Summary": "all",
            "X-Retain-Images": "none",
            "X-Max-Tokens": "6000",
            "X-Robots-Txt": "sifty-crawl",
        }
        if self.settings.jina_api_key:
            headers["Authorization"] = f"Bearer {self.settings.jina_api_key}"
        try:
            # POST preserves hash routes and avoids putting the caller's URL in Reader's path.
            with self.client.stream(
                "POST",
                self.settings.reader_url.rstrip("/") + "/",
                json={"url": url},
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ReaderError(f"Jina Reader answered HTTP {response.status_code}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ReaderError("Jina Reader response is too large")
                    if time.monotonic() >= deadline:
                        raise ReaderError("Crawl time budget exhausted")
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
        except httpx.HTTPError as exc:
            # Never return upstream exception text: it can contain the key or source URL.
            raise ReaderError(f"Jina Reader request failed: {type(exc).__name__}") from exc
        except (ValueError, UnicodeError) as exc:
            raise ReaderError("Jina Reader returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("code", 200) != 200:
            raise ReaderError("Jina Reader could not read the page")
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("content"), str):
            raise ReaderError("Jina Reader returned no page content")
        status = data.get("httpStatus", 200)
        if not isinstance(status, int) or not 200 <= status < 400 or not data["content"].strip():
            raise ReaderError("The source page returned no usable content")
        source = data.get("url", url)
        if not isinstance(source, str):
            raise ReaderError("Jina Reader returned an invalid source URL")
        source = self.validate(source)
        title = data.get("title", "")
        if not isinstance(title, str):
            title = ""
        links = data.get("links", {})
        pairs = (
            links.items() if isinstance(links, dict) else links if isinstance(links, list) else []
        )
        parsed = {}
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            text, href = pair
            if not isinstance(text, str) or not isinstance(href, str):
                continue
            try:
                target = normalize_url(urljoin(source, href))
            except PromptError:
                continue
            parsed.setdefault(target, Link(text[:256], target))
        return Page(source, title[:512], data["content"][:80_000], tuple(parsed.values()))
