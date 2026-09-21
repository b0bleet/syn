import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helpers import CharacterTokenizer, FixedBackend

from syn.backends import FeatureResult
from syn.config import Settings
from syn.head import AttentionHead
from syn.prompt import PromptBuilder
from syn.schema import ScoreRequest
from syn.scoring import Scorer

HIDDEN = 8


class FeatureBackend:
    def features(self, prefix_ids, spans):
        rng = np.random.default_rng(len(prefix_ids))
        ctx = rng.normal(size=(len(prefix_ids), HIDDEN)).astype(np.float16)
        opts = np.stack([rng.normal(size=HIDDEN) for _ in spans]).astype(np.float16)
        return FeatureResult(ctx, opts, compute_ms=1.0, wait_ms=0.0)

    def close(self):
        pass


def scorer_with(settings, head_temperature):
    torch.manual_seed(0)
    builder = PromptBuilder(CharacterTokenizer(), settings.max_prompt_tokens)
    backend = FeatureBackend() if settings.readout == "head" else FixedBackend()
    head = AttentionHead(HIDDEN, 4) if settings.readout == "head" else None
    return Scorer(settings, builder, backend, "abc", head, "sha", head_temperature)


def test_the_stored_head_temperature_shapes_the_probabilities(request_data):
    request = ScoreRequest.model_validate(request_data)
    settings = Settings(readout="head", head_path="unused.safetensors")
    raw = scorer_with(settings, None).score(request)
    tempered = scorer_with(settings, 2.5).score(request)
    assert raw.temperature == 1.0 and tempered.temperature == 2.5
    logits = [s.log_probability for s in raw.scores]
    assert logits == [s.log_probability for s in tempered.scores]  # logits are untouched
    scaled = [math.exp(x / 2.5) for x in logits]
    expected = [x / sum(scaled) for x in scaled]
    for score, p in zip(tempered.scores, expected, strict=True):
        assert score.probability == pytest.approx(p)
    assert tempered.confidence <= raw.confidence  # a temperature above one flattens


def test_an_explicit_setting_overrides_the_head_temperature(request_data):
    request = ScoreRequest.model_validate(request_data)
    explicit = Settings(readout="head", head_path="unused.safetensors", temperature=1.0)
    assert scorer_with(explicit, 2.5).score(request).temperature == 1.0
    # Other readouts never see a head temperature.
    letters = scorer_with(Settings(), 2.5).score(request)
    assert letters.temperature == 1.0
