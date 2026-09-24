"""Images to score: decoded from a data: URI, or fetched from a public http(s) URL.

A URL fetch runs on the scoring server, so it is guarded against server-side request forgery:
only http and https; only public addresses, checked on every resolved address before connecting
and again on the address actually connected to (a DNS answer that changes between the two is
caught); a few redirects, each checked the same way; at most MAX_BYTES; and a timeout.

Decoding refuses images over MAX_DECODED_PIXELS before reading pixel data, applies the EXIF
orientation, converts to RGB, and scales the image down to at most `max_pixels`, which bounds
how many image tokens a request can add.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import ipaddress
import re
import socket
import threading
from collections import OrderedDict
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = 10 * 1024 * 1024
# Base64 of MAX_BYTES, plus room for the data: header.
MAX_REFERENCE_CHARS = 4 * MAX_BYTES // 3 + 1024
MAX_DECODED_PIXELS = 50_000_000
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 10.0
# Image hosts such as Wikimedia refuse clients that don't say who they are.
HEADERS = {
    "User-Agent": "sifty-image-fetch/1.0 (+https://sifty.dev; fetches one image a caller asked about)",
    "Accept": "image/*",
}
DATA_URI = re.compile(r"^data:image/[a-z0-9.+-]+;base64,", re.IGNORECASE)


class ImageError(ValueError):
    pass


def public(address: str) -> bool:
    """Whether an address is on the public internet: not private, loopback, link-local, etc."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def resolve(host: str, port: int) -> list[str]:
    return [info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def decode_data_uri(reference: str) -> bytes:
    if not DATA_URI.match(reference):
        raise ImageError("An image data: URI must look like data:image/png;base64,...")
    payload = reference.split(",", 1)[1]
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("The image data: URI is not valid base64") from exc
    if len(data) > MAX_BYTES:
        raise ImageError(f"The image is over {MAX_BYTES // (1024 * 1024)} MB")
    return data


def fetch(url: str, client: httpx.Client | None = None, resolver=resolve) -> bytes:
    """The body of a public http(s) URL, following at most MAX_REDIRECTS checked redirects."""
    own = client is None
    client = client or httpx.Client(
        timeout=TIMEOUT_SECONDS, follow_redirects=False, headers=HEADERS
    )
    try:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ImageError("An image URL must be http or https")
            port = parts.port or (443 if parts.scheme == "https" else 80)
            try:
                addresses = resolver(parts.hostname, port)
            except OSError as exc:
                raise ImageError(f"Could not resolve the image host {parts.hostname}") from exc
            if not addresses or not all(public(address) for address in addresses):
                raise ImageError("The image URL must point to a public address")
            with client.stream("GET", url, follow_redirects=False) as response:
                stream = response.extensions.get("network_stream")
                peer = stream.get_extra_info("server_addr") if stream is not None else None
                if peer is not None and not public(peer[0]):
                    raise ImageError("The image URL must point to a public address")
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                if response.status_code != 200:
                    raise ImageError(f"The image URL answered HTTP {response.status_code}")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ImageError(f"The image is over {MAX_BYTES // (1024 * 1024)} MB")
                    chunks.append(chunk)
                return b"".join(chunks)
        raise ImageError("The image URL redirected too many times")
    except httpx.HTTPError as exc:
        raise ImageError(f"Could not fetch the image: {exc.__class__.__name__}") from exc
    finally:
        if own:
            client.close()


def decode(data: bytes, max_pixels: int):
    """An RGB PIL image, upright, scaled down to at most `max_pixels`."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > MAX_DECODED_PIXELS:
            raise ImageError("The image has too many pixels")
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("The image could not be decoded") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def load(reference: str, max_pixels: int, client: httpx.Client | None = None, resolver=resolve):
    """The image a request refers to, as an RGB PIL image."""
    reference = reference.strip()
    if len(reference) > MAX_REFERENCE_CHARS:
        raise ImageError(f"The image is over {MAX_BYTES // (1024 * 1024)} MB")
    if reference[:5].lower() == "data:":
        data = decode_data_uri(reference)
    else:
        data = fetch(reference, client, resolver)
    return decode(data, max_pixels)


# A few recently loaded images, so the questions of one System One request (each scored as its
# own request) fetch and decode their shared image once. Keyed by a digest, not the reference.
CACHE_SIZE = 8
_cache: OrderedDict[tuple[str, int], object] = OrderedDict()
_cache_lock = threading.Lock()


def load_cached(reference: str, max_pixels: int):
    key = (hashlib.sha256(reference.strip().encode()).hexdigest(), max_pixels)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    image = load(reference, max_pixels)
    with _cache_lock:
        _cache[key] = image
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return image


class ImageInputs:
    """Qwen's model inputs for one prompt with images, without Qwen's processor class.

    That class also builds a video processor, which needs torchvision. This does what it does for
    images: the Pillow image processor turns each image into patches and a (t, h, w) grid, the
    template's one image placeholder is repeated once per merged patch, and `mm_token_type_ids`
    marks those tokens (1) so the model can place them in 2D.
    """

    def __init__(self, image_processor, tokenizer, image_token: str = "<|image_pad|>"):
        self.image_processor, self.tokenizer, self.image_token = (
            image_processor,
            tokenizer,
            image_token,
        )
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def __call__(self, text: list[str], images: list, return_tensors: str = "pt") -> dict:
        import torch

        [rendered] = text
        parts = rendered.split(self.image_token)
        if len(parts) != len(images) + 1:
            raise ValueError("The prompt needs one image placeholder per image")
        pixels = self.image_processor(images=images, return_tensors="pt")
        grid = pixels["image_grid_thw"]
        merged = self.image_processor.merge_size**2
        expanded = parts[0] + "".join(
            self.image_token * int(grid[i].prod() // merged) + parts[i + 1]
            for i in range(len(images))
        )
        input_ids = torch.tensor([self.tokenizer.encode(expanded, add_special_tokens=False)])
        return {
            "input_ids": input_ids,
            "pixel_values": pixels["pixel_values"],
            "image_grid_thw": grid,
            "mm_token_type_ids": (input_ids == self.image_token_id).int(),
        }
