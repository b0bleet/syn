import base64
import io
import math

import pytest
import torch
from fastapi.testclient import TestClient
from helpers import CharacterTokenizer, ContentBackend
from PIL import Image

from syn.api import create_app
from syn.config import Settings
from syn.prompt import IMAGE_CONTEXT, PromptBuilder, PromptError
from syn.schema import ClassifyRequest, ScoreRequest
from syn.scoring import Scorer
from syn.shorthand import IMAGE_QUESTION

PAD = "\x01"
PATCHES = 4


def image_uri(width=8, height=8) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (0, 128, 255)).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


URI = image_uri()


class ImageTokenizer(CharacterTokenizer):
    """Writes Qwen's image placeholder where a user turn holds an image."""

    def apply_chat_template(self, messages, **kwargs):
        user = messages[1]["content"]
        if isinstance(user, list):
            user = "".join(
                "<|vision_start|><|image_pad|><|vision_end|>"
                if part["type"] == "image"
                else part["text"]
                for part in user
            )
        return super().apply_chat_template(
            [messages[0], {"role": "user", "content": user}], **kwargs
        )


class FakeProcessor:
    """Expands the placeholder into one pad token per patch, as Qwen's processor does."""

    def __init__(self):
        self.sizes = []

    def __call__(self, text, images, return_tensors):
        [rendered], [image] = text, images
        self.sizes.append(image.size)
        ids = [ord(c) for c in rendered.replace("<|image_pad|>", PAD * PATCHES)]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids)),
            "pixel_values": torch.zeros(PATCHES * 4, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }


def request(**update) -> ScoreRequest:
    fields = {
        "question": "Which animal is shown?",
        "options": [{"id": "cat", "text": "cat"}, {"id": "dog", "text": "dog"}],
        "image": URI,
    }
    return ScoreRequest.model_validate({**fields, **update})


def test_an_image_prompt_puts_the_image_first_and_keeps_the_label_tokens():
    processor = FakeProcessor()
    builder = PromptBuilder(ImageTokenizer(), 8192, "json", processor)
    prompt = builder.prepare(request())
    text = "".join(chr(i) for i in prompt.input_ids)
    assert text.index(PAD * PATCHES) < text.index('"question"')
    assert IMAGE_CONTEXT in text  # no text given: the image is the state
    assert set(prompt.vision) == {"pixel_values", "image_grid_thw"}
    as_text = builder.prepare(request(image=None, context="a photo"))
    assert as_text.vision is None
    assert prompt.labels == as_text.labels == ["A", "B"]
    assert prompt.label_token_ids == as_text.label_token_ids
    # A caption goes into the context alongside the image.
    captioned = "".join(
        chr(i) for i in builder.prepare(request(context="taken at night")).input_ids
    )
    assert "taken at night" in captioned and IMAGE_CONTEXT not in captioned


def test_images_are_scaled_to_the_pixel_budget():
    processor = FakeProcessor()
    builder = PromptBuilder(ImageTokenizer(), 8192, "json", processor, image_max_pixels=64 * 64)
    builder.prepare(request(image=image_uri(400, 100)))
    [(width, height)] = processor.sizes
    assert width * height <= 64 * 64


def test_images_are_refused_without_a_processor_and_when_bad_or_too_long():
    with pytest.raises(PromptError, match="does not accept images"):
        PromptBuilder(ImageTokenizer(), 8192).prepare(request())
    builder = PromptBuilder(ImageTokenizer(), 8192, "json", FakeProcessor())
    with pytest.raises(PromptError, match="could not be decoded"):
        builder.prepare(request(image="data:image/png;base64," + base64.b64encode(b"x").decode()))
    with pytest.raises(PromptError, match="public address"):
        builder.prepare(request(image="http://127.0.0.1/cat.png"))
    short = PromptBuilder(ImageTokenizer(), 50, "json", FakeProcessor())
    with pytest.raises(PromptError, match="limit is 50"):
        short.prepare(request())


def test_images_need_the_letters_readout():
    tokenizer = ImageTokenizer()
    settings = Settings(readout="pmi")
    builder = PromptBuilder(tokenizer, 8192, "json", FakeProcessor())
    scorer = Scorer(settings, builder, ContentBackend(tokenizer, {}))
    with pytest.raises(PromptError, match="letters readout"):
        scorer.score(request())


def test_a_request_needs_text_or_an_image():
    with pytest.raises(ValueError, match="context, an image"):
        request(image=None)
    assert request(context="").image == URI
    labels = ["cat", "dog"]
    with pytest.raises(ValueError, match="input text, an image"):
        ClassifyRequest(labels=labels)
    with pytest.raises(ValueError, match="not a batch"):
        ClassifyRequest(input=["a", "b"], labels=labels, image=URI)
    assert ClassifyRequest(labels=labels, image=URI).input is None


def client(processor=True) -> TestClient:
    tokenizer = ImageTokenizer()
    builder = PromptBuilder(tokenizer, 8192, "json", FakeProcessor() if processor else None)
    backend = ContentBackend(tokenizer, {"cat": math.log(0.8), "dog": math.log(0.2)})
    return TestClient(create_app(Settings(), Scorer(Settings(), builder, backend, "abc")))


def test_every_route_accepts_an_image():
    with client() as api:
        body = api.post("/", json={"image": URI, "labels": ["dog", "cat"]}).json()
        assert [r["label"] for r in body["results"]] == ["cat"]
        captioned = api.post("/", json={"image": URI, "input": "a pet", "labels": ["cat", "dog"]})
        assert captioned.status_code == 200
        batch = {"image": URI, "input": ["a", "b"], "labels": ["cat", "dog"]}
        assert api.post("/", json=batch).status_code == 422
        assert api.get("/", params={"labels": "cat,dog", "image": URI}).text == "cat\n"
        assert api.get("/", params={"labels": "cat", "image": URI}).status_code == 422
        scored = api.post("/v1/score", json=request().model_dump(exclude_none=True)).json()
        assert scored["selected_option_id"] == "cat"
        one = {
            "state": "",
            "model": "syn-latest",
            "image": URI,
            "questions": {
                "animal": {
                    "type": "choice",
                    "instructions": "Which animal is shown?",
                    "criteria": {"cat": None, "dog": None},
                }
            },
        }
        answer = api.post("/v1/systemone", json=one)
        assert answer.status_code == 200, answer.text
        assert answer.json()["answers"]["animal"]["choice"] == "cat"


def test_the_default_question_names_the_image():
    tokenizer = ImageTokenizer()
    builder = PromptBuilder(tokenizer, 8192, "json", FakeProcessor())
    seen = []

    class Recording(ContentBackend):
        def score(self, prompts):
            seen.extend(prompts)
            return super().score(prompts)

    backend = Recording(tokenizer, {"cat": math.log(0.8), "dog": math.log(0.2)})
    with TestClient(create_app(Settings(), Scorer(Settings(), builder, backend, "abc"))) as api:
        api.post("/", json={"image": URI, "labels": ["cat", "dog"]})
    assert IMAGE_QUESTION in "".join(chr(i) for i in seen[0].input_ids)


def test_a_server_without_images_says_so():
    with client(processor=False) as api:
        refused = api.post("/", json={"image": URI, "labels": ["cat", "dog"]})
        assert refused.status_code == 422
        assert "does not accept images" in refused.json()["detail"]
