"""Per-thread / per-channel model picks via @@model markers (claude_handler)."""

import asyncio
import json

import pytest

import claude_handler
import session_store
from claude_handler import (
    ClaudeHandler, _MODEL_NAME_RE, _ProgressTracker, _model_context, _model_window,
    _parse_model_markers,
)


class TestParseModelMarkers:
    def test_strips_marker_lines_and_lowercases(self):
        text, picks = _parse_model_markers("Switching now.\n@@model Thread Fable\nBye.")
        assert text == "Switching now.\nBye."
        assert picks == [("thread", "fable")]

    def test_channel_default_and_backticked_marker(self):
        text, picks = _parse_model_markers("`@@model channel default`")
        assert text == ""
        assert picks == [("channel", "default")]

    def test_no_marker_is_untouched(self):
        assert _parse_model_markers("plain reply") == ("plain reply", [])

    def test_inline_mention_is_not_a_marker(self):
        text, picks = _parse_model_markers("say @@model thread fable to switch")
        assert picks == [] and text.startswith("say @@model")


@pytest.mark.parametrize("name", [
    "opus", "sonnet", "haiku", "fable", "best", "opusplan", "opus[1m]", "sonnet[1m]",
    "claude-opus-5-5", "claude-sonnet-4-5[1m]",
])
def test_valid_model_names(name):
    assert _MODEL_NAME_RE.match(name)


@pytest.mark.parametrize("name", ["fable5", "gpt-4", "opus[2m]", "claude", "", "opus plan"])
def test_invalid_model_names(name):
    assert not _MODEL_NAME_RE.match(name)


def test_model_window_known_and_unknown():
    assert _model_window("haiku") == 200_000
    assert _model_window("opus[1m]") == 1_000_000
    assert _model_window("opusplan") == 1_000_000          # "opus" substring
    assert _model_window("best") is None                     # unknown family: can't judge
    assert _model_context("best") == claude_handler._DEFAULT_CONTEXT  # unchanged fallback


def test_tracker_exposes_ctx_last():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"content": [], "usage": {
        "input_tokens": 1000, "cache_read_input_tokens": 449_000, "output_tokens": 5}}})
    assert t.ctx_last == 450_000
