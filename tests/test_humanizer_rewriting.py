"""Meaningful checks of output validity and provenance-bound rewrite resume."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from humanizers.rewriting import (
    _load_model, _rewrite_one, dipper_prompt, generation_status, read_jsonl, run_humanizers,
)


def inputs(tmp_path):
    cohort = tmp_path / "cohort.jsonl"
    sources = [
        dict(id="test-a", source_id="a", source_group_id="g-a", domain="Essay",
             generator="ChatGPT", split="test", human="A human passage.",
             machine="A machine passage."),
        dict(id="cal-b", source_id="b", source_group_id="g-b", domain="Reuters",
             generator="Claude", split="calibration", human="Calibration human.",
             machine="Calibration machine."),
    ]
    cohort.write_text("".join(json.dumps(row) + "\n" for row in sources))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 42, "humanizers": [{"name": "identity", "kind": "identity"}]}))
    return cohort, config


def test_identity_is_exact_and_excludes_calibration_and_resumes_without_duplicates(tmp_path):
    cohort, config = inputs(tmp_path)
    output = tmp_path / "output"
    first = run_humanizers(cohort, config, output, device="cpu")
    assert first["expected_rewrites"] == 1
    assert first["status_counts"] == {"ok": 1}
    rows = read_jsonl(output / "rewrites.jsonl")
    assert rows[0]["original"] == rows[0]["modified"] == "A machine passage."
    assert rows[0]["source_id"] == "test-a"
    assert rows[0]["source_group_id"] == "g-a"
    assert rows == read_jsonl(output / "pairs.jsonl")
    before = (output / "rewrites.jsonl").read_bytes()
    second = run_humanizers(cohort, config, output, device="cpu")
    assert second["resumed_completed_rewrites"] == 1
    assert (output / "rewrites.jsonl").read_bytes() == before


def test_resume_refuses_changed_text_or_generation_configuration(tmp_path):
    cohort, config = inputs(tmp_path)
    output = tmp_path / "output"
    run_humanizers(cohort, config, output, device="cpu")
    original = config.read_text()
    config.write_text(original.replace("42", "43"))
    with pytest.raises(ValueError, match="Resume configuration/cohort/device differs"):
        run_humanizers(cohort, config, output, device="cpu")
    config.write_text(original)
    cohort.write_text(cohort.read_text().replace("machine passage", "different passage"))
    with pytest.raises(ValueError, match="Resume configuration/cohort/device differs"):
        run_humanizers(cohort, config, output, device="cpu")


def test_empty_and_token_limited_generations_are_not_successful_pairs():
    assert generation_status("", 1, 512, True) == "generation_error"
    assert generation_status("A partial passage", 512, 512, False) == "truncated"
    assert generation_status("A finished passage.", 512, 512, True) == "ok"
    assert generation_status("A finished passage.", 20, 512, True) == "ok"


@pytest.mark.parametrize('requested,current,expected', [('cuda', 1, 1), ('cuda:0', 1, 0)])
def test_cuda_alias_and_explicit_gpu_select_an_integer_without_downloading(
        monkeypatch, requested, current, expected):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: current)

    def set_device(index):
        # Reproduce the strict API which rejected torch.device('cuda').
        assert isinstance(index, int)
        selected.append(index)

    monkeypatch.setattr(torch.cuda, 'set_device', set_device)
    tokenizer = object()
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)

    class Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    model = Model()

    def from_pretrained(*args, **kwargs):
        assert kwargs['torch_dtype'] == torch.float16
        return model

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', from_pretrained)
    actual_tokenizer, actual_model, actual_device = _load_model(
        {'kind': 'causal', 'model_id': 'test/no-download', 'revision': '0'*40}, requested)
    assert selected == [expected]
    assert actual_device == torch.device('cuda', expected) == model.device
    assert actual_model is model and actual_tokenizer is tokenizer


def test_dipper_prompt_uses_inverse_codes_and_does_not_drop_text():
    assert dipper_prompt("First sentence.\nSecond sentence.", 60, 0) == (
        "lexical = 40, order = 100 <sent> First sentence. Second sentence. </sent>")
    with pytest.raises(ValueError, match="DIPPER diversity"):
        dipper_prompt("Passage.", 55, 0)


def test_oversized_prompt_is_rejected_before_generation_without_truncation():
    import torch
    source = dict(id="a", source_group_id="g", domain="Essay",
                  generator="Claude", machine="A whole input passage.")
    system = dict(name="rewrite", kind="causal", prompt="Rewrite carefully.",
                  max_input_tokens=2, max_new_tokens=8)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "A full formatted prompt"

        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    class Model:
        config = SimpleNamespace(max_position_embeddings=2048)

        def generate(self, **kwargs):
            pytest.fail("An oversized input must never reach generation")

    row = _rewrite_one(source, system, 42, Tokenizer(), Model(), torch.device("cpu"))
    assert row["status"] == "input_too_long"
    assert row["modified"] == ""
    assert row["metadata"]["input_tokens"] == 3


def test_causal_output_does_not_include_prompt_and_eos_at_limit_is_complete():
    import torch
    source = dict(id="a", source_group_id="g", domain="Essay",
                  generator="Claude", machine="An input passage.")
    system = dict(name="rewrite", kind="causal", prompt="Rewrite carefully.",
                  max_input_tokens=10, max_new_tokens=3)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, *args, **kwargs):
            return "A full formatted prompt"

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[11, 12]])}

        def decode(self, tokens, **kwargs):
            assert tokens == [22, 23, 9]
            return "A rewritten passage."

    class Model:
        config = SimpleNamespace(max_position_embeddings=2048)
        generation_config = SimpleNamespace(eos_token_id=[8, 9])

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[11, 12, 22, 23, 9]])

    row = _rewrite_one(source, system, 42, Tokenizer(), Model(), torch.device("cpu"))
    assert row["modified"] == "A rewritten passage."
    assert row["status"] == "ok"
    assert row["metadata"]["output_tokens"] == 3
    assert row["metadata"]["finished_eos"] is True
