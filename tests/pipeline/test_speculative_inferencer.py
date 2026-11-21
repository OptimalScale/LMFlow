from types import SimpleNamespace

import torch

from lmflow.pipeline.inferencer import SpeculativeInferencer


class _FakeBackendModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor, return_dict: bool = True):
        self.calls += 1
        seq_len = input_ids.shape[1]
        logits = self._logits.expand(input_ids.shape[0], seq_len, -1).clone()
        return SimpleNamespace(logits=logits)


class _FakeHFDecoderModel:
    def __init__(self, logits: torch.Tensor):
        self._backend = _FakeBackendModel(logits)
        self.inference_called = False

    def get_backend_model(self):
        return self._backend

    def inference(self, input_ids: torch.Tensor, **kwargs):
        self.inference_called = True
        raise AssertionError("SpeculativeInferencer.autoregressive_sampling must not call HF generate()")

    def encode(self, text: str, return_tensors: str = "pt"):
        return torch.tensor([[0]], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor):
        return ",".join(str(int(t)) for t in token_ids)


def test_autoregressive_sampling_uses_raw_backend_logits():
    raw_logits = torch.tensor([[[2.0, 1.0, 0.0]]])
    model = _FakeHFDecoderModel(raw_logits)
    inferencer = SpeculativeInferencer.__new__(SpeculativeInferencer)

    result = inferencer.autoregressive_sampling(
        input_ids=torch.tensor([[0]], dtype=torch.long),
        model=model,
        temperature=1.0,
        num_new_tokens=2,
    )

    expected_prob = torch.softmax(raw_logits[:, -1, :], dim=-1)
    assert not model.inference_called
    assert model.get_backend_model().calls == 2
    assert result["sequence"].shape == (1, 3)
    for token_info in result["new_tokens"]:
        assert torch.allclose(token_info["all_prob"], expected_prob)


def test_score_to_prob_does_not_mutate_scores_in_place():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    original = scores.clone()

    prob_first = SpeculativeInferencer.score_to_prob(scores, temperature=0.5)
    prob_second = SpeculativeInferencer.score_to_prob(scores, temperature=0.5)

    assert torch.equal(scores, original)
    assert torch.allclose(prob_first, prob_second)


def test_speculative_inference_forwards_temperature_to_draft_model():
    raw_logits = torch.tensor([[[2.0, 1.0, 0.0]]])
    target_model = _FakeHFDecoderModel(raw_logits)
    draft_model = _FakeHFDecoderModel(raw_logits)

    inferencer = SpeculativeInferencer.__new__(SpeculativeInferencer)
    inferencer.inferencer_args = SimpleNamespace(device="cpu")

    seen_temperatures = []
    orig_ar = inferencer.autoregressive_sampling

    def spy_ar(input_ids, model, temperature=0.0, num_new_tokens=5):
        seen_temperatures.append(temperature)
        return orig_ar(input_ids=input_ids, model=model, temperature=temperature, num_new_tokens=num_new_tokens)

    inferencer.autoregressive_sampling = spy_ar
    out = inferencer.inference(
        model=target_model,
        draft_model=draft_model,
        input="hello",
        temperature=0.7,
        gamma=2,
        max_new_tokens=2,
    )

    assert isinstance(out, str)
    assert seen_temperatures and all(t == 0.7 for t in seen_temperatures)
    assert not draft_model.inference_called
