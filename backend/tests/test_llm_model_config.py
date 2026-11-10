import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from llm_model_config import camel_model_config  # noqa: E402


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-6-luna", {"reasoning_effort": "none"}),
        ("gpt-6-sol", {"reasoning_effort": "none"}),
        ("openai/gpt-6-luna", {"reasoning_effort": "none"}),
        ("gpt-5", None),
        ("gpt-4o-mini", None),
        ("qwen-plus", None),
        ("", None),
    ],
)
def test_default_config(monkeypatch, model, expected):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert camel_model_config(model) == expected


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    assert camel_model_config("gpt-4o-mini") == {"reasoning_effort": "low"}
