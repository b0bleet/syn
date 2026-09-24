import base64
import io

import httpx
import pytest
from PIL import Image

from syn import images
from syn.images import ImageError


def png(width=8, height=6, mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), (255, 0, 0, 255)[: len(mode)]).save(buffer, "PNG")
    return buffer.getvalue()


def data_uri(data: bytes, kind="png") -> str:
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def public_resolver(host, port):
    return ["93.184.216.34"]


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("93.184.216.34", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("192.168.0.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),  # carrier-grade NAT
        ("::1", False),
        ("fc00::1", False),
        ("fe80::1%eth0", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback
        ("224.0.0.1", False),
        ("not an address", False),
    ],
)
def test_only_public_addresses_are_allowed(address, expected):
    assert images.public(address) is expected


def test_a_data_uri_decodes_to_an_rgb_image():
    image = images.load(data_uri(png(8, 6, "RGBA")), max_pixels=1_000_000)
    assert image.mode == "RGB" and image.size == (8, 6)


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("data:text/plain;base64,aGk=", "must look like"),
        ("data:image/png;base64,not base64!", "not valid base64"),
        (data_uri(b"not an image"), "could not be decoded"),
    ],
)
def test_bad_data_uris_are_refused(reference, message):
    with pytest.raises(ImageError, match=message):
        images.load(reference, max_pixels=1_000_000)


def test_large_images_are_refused_and_big_ones_scaled_down(monkeypatch):
    image = images.load(data_uri(png(400, 100)), max_pixels=10_000)
    assert image.size[0] * image.size[1] <= 10_000
    assert image.size[0] / image.size[1] == pytest.approx(4, rel=0.05)
    monkeypatch.setattr(images, "MAX_BYTES", 10)
    with pytest.raises(ImageError, match="over"):
        images.load(data_uri(png()), max_pixels=10_000)
    monkeypatch.undo()
    monkeypatch.setattr(images, "MAX_DECODED_PIXELS", 100)
    with pytest.raises(ImageError, match="too many pixels"):
        images.load(data_uri(png(20, 20)), max_pixels=10_000)


def test_a_public_url_is_fetched():
    body = png()
    fetched = client(lambda request: httpx.Response(200, content=body))
    image = images.load("https://example.com/cat.png", 1_000_000, fetched, public_resolver)
    assert image.size == (8, 6)


@pytest.mark.parametrize(
    "url", ["ftp://example.com/a.png", "file:///etc/passwd", "https:///nohost", "cat.png"]
)
def test_only_http_urls_are_fetched(url):
    never = client(lambda request: pytest.fail("nothing must be fetched"))
    with pytest.raises(ImageError, match="http or https"):
        images.fetch(url, never, public_resolver)


def test_private_hosts_are_refused_before_connecting():
    never = client(lambda request: pytest.fail("a private host must not be contacted"))
    for addresses in (["10.0.0.5"], ["93.184.216.34", "127.0.0.1"], []):
        with pytest.raises(ImageError, match="public address"):
            images.fetch("http://internal.example/a.png", never, lambda h, p, a=addresses: a)


def test_redirects_are_checked_and_limited():
    def resolver(host, port):
        return ["10.0.0.9"] if host == "metadata.internal" else ["93.184.216.34"]

    to_private = client(
        lambda request: httpx.Response(302, headers={"location": "http://metadata.internal/x"})
    )
    with pytest.raises(ImageError, match="public address"):
        images.fetch("https://example.com/a.png", to_private, resolver)

    looping = client(lambda request: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(ImageError, match="too many"):
        images.fetch("https://example.com/a.png", looping, resolver)

    hops = iter(["/b", "/c"])

    def two_hops(request):
        location = next(hops, None)
        if location:
            return httpx.Response(301, headers={"location": location})
        return httpx.Response(200, content=b"ok")

    assert images.fetch("https://example.com/a", client(two_hops), resolver) == b"ok"


def test_failed_and_oversized_downloads_are_refused(monkeypatch):
    missing = client(lambda request: httpx.Response(404))
    with pytest.raises(ImageError, match="HTTP 404"):
        images.fetch("https://example.com/a.png", missing, public_resolver)

    monkeypatch.setattr(images, "MAX_BYTES", 5)
    large = client(lambda request: httpx.Response(200, content=b"0123456789"))
    with pytest.raises(ImageError, match="over"):
        images.fetch("https://example.com/a.png", large, public_resolver)

    def unreachable(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ImageError, match="Could not fetch"):
        images.fetch("https://example.com/a.png", client(unreachable), public_resolver)


def test_image_inputs_repeat_the_placeholder_once_per_merged_patch():
    import torch

    class Pixels:
        merge_size = 2

        def __call__(self, images, return_tensors):
            # A 1 x 4 x 6 patch grid: 24 patches, 6 once merged 2 x 2.
            return {"pixel_values": torch.zeros(24, 3), "image_grid_thw": torch.tensor([[1, 4, 6]])}

    class Tokens:
        def convert_tokens_to_ids(self, token):
            return {"<|image_pad|>": 7}[token]

        def encode(self, text, add_special_tokens=False):
            ids, rest = [], text
            while rest:
                if rest.startswith("<|image_pad|>"):
                    ids.append(7)
                    rest = rest[len("<|image_pad|>") :]
                else:
                    ids.append(100 + ord(rest[0]))
                    rest = rest[1:]
            return ids

    inputs = images.ImageInputs(Pixels(), Tokens())
    out = inputs(text=["ab<|image_pad|>c"], images=[object()])
    assert out["input_ids"][0].tolist() == [197, 198, 7, 7, 7, 7, 7, 7, 199]
    assert out["mm_token_type_ids"][0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert out["image_grid_thw"].tolist() == [[1, 4, 6]]
    with pytest.raises(ValueError, match="one image placeholder per image"):
        inputs(text=["no placeholder"], images=[object()])
