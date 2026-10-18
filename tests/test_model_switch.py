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


class FakeSlack:
    async def auth_test(self): return {"user_id": "UBOT"}
    async def conversations_list(self, **kw): return {"channels": []}  # raw-ID keys resolve w/o Slack
    async def conversations_replies(self, **kw): return {"messages": []}


class SpyHandler(ClaudeHandler):
    """Scripted _run_claude: records (cmd, prompt), pops the next reply."""
    def __init__(self, *a, replies, **k):
        super().__init__(*a, **k)
        self.runs: list[dict] = []
        self._scripted = list(replies)

    async def _run_claude(self, cmd, prompt, cwd=None, thread_ts=None, channel=None, progress_cb=None):
        self.runs.append({"cmd": cmd, "prompt": prompt})
        return self._scripted.pop(0)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated projects.json + sessions.json, CLAUDE_MODEL unset, --resume path available."""
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    projects = tmp_path / "projects.json"
    monkeypatch.setattr(claude_handler, "PROJECTS_CONFIG", projects)
    jsonl = tmp_path / "sess.jsonl"
    jsonl.write_text("{}")
    monkeypatch.setattr(claude_handler, "_jsonl_path", lambda cwd, sid: jsonl)
    return projects


def make(tmp_path, env, replies, mapping=None):
    if mapping is not None:
        env.write_text(json.dumps(mapping))
    h = SpyHandler(FakeSlack(), store_path=tmp_path / "sessions.json", replies=replies)
    asyncio.run(h.initialize())
    return h


def with_thread(h, tmp_path, thread_ts="T1", sid="sid-1"):
    """Register a resumable thread (cwd exists so the self-heal path stays quiet)."""
    h._sessions[thread_ts] = sid
    h._thread_config[thread_ts] = (str(tmp_path), None)
    return h


def model_flag(cmd):
    return cmd[cmd.index("--model") + 1] if "--model" in cmd else None


def system_prompt(cmd):
    return cmd[cmd.index("--append-system-prompt") + 1]


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


class TestBuildCmdAndPrompt:
    def test_model_flag_and_runs_on_sentence(self):
        cmd = ClaudeHandler._build_cmd(resume="sid", model="fable")
        assert model_flag(cmd) == "fable"
        assert system_prompt(cmd).endswith("This session currently runs on: fable.")

    def test_no_model_means_no_flag_and_default_sentence(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_MODEL", "opus")  # _build_cmd no longer reads the env itself
        cmd = ClaudeHandler._build_cmd(session_id="sid")
        assert "--model" not in cmd
        assert system_prompt(cmd).endswith("This session currently runs on: the default model.")

    def test_prompt_teaches_marker(self):
        p = ClaudeHandler._FLOW_B_SYSTEM_PROMPT
        assert "@@model thread <model>" in p and "@@model channel" in p and "default" in p
        assert "already runs on" in p


class TestEffectiveModel:
    def test_precedence_thread_channel_env_none(self, tmp_path, env, monkeypatch):
        h = make(tmp_path, env, replies=[], mapping={"C1": {"path": str(tmp_path), "model": "sonnet"}})
        assert h._effective_model("T1", "C1") == "sonnet"            # channel pick
        h._thread_model["T1"] = "fable"
        assert h._effective_model("T1", "C1") == "fable"             # thread pick wins
        assert h._effective_model("T2", "C_OTHER") is None           # nothing anywhere
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        assert h._effective_model("T2", "C_OTHER") == "opus"         # env is the global default

    def test_resolved_config_carries_model_and_key(self, tmp_path, env):
        h = make(tmp_path, env, replies=[], mapping={"C1": str(tmp_path)})
        assert h._channel_id_to_project["C1"]["model"] is None
        assert h._channel_id_to_project["C1"]["key"] == "C1"


class TestRunsUseEffectiveModel:
    def test_handle_message_passes_env_model_when_no_pick(self, tmp_path, env, monkeypatch):
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        h = make(tmp_path, env, replies=["ok"])
        asyncio.run(h.handle_message("C1", "T1", "hi"))
        assert model_flag(h.runs[0]["cmd"]) == "opus" and "--session-id" in h.runs[0]["cmd"]

    def test_new_thread_starts_on_channel_pick(self, tmp_path, env):
        h = make(tmp_path, env, replies=["ok"], mapping={"C1": {"path": str(tmp_path), "model": "fable"}})
        asyncio.run(h.handle_message("C1", "T1", "hi"))
        assert model_flag(h.runs[0]["cmd"]) == "fable"

    def test_resume_and_scrape_use_thread_pick(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["ok", "scraped"]), tmp_path)
        h._thread_model["T1"] = "haiku"
        asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert model_flag(h.runs[0]["cmd"]) == "haiku" and "--resume" in h.runs[0]["cmd"]
        h._thread_config["T9"] = (str(tmp_path), None)   # no session -> scrape path
        h._thread_model["T9"] = "haiku"
        asyncio.run(h.handle_thread_reply("C1", "T9", "hi"))
        assert model_flag(h.runs[1]["cmd"]) == "haiku" and "--session-id" in h.runs[1]["cmd"]


class TestPersistence:
    def test_initialize_restores_model_and_ctx(self, tmp_path, env):
        store = tmp_path / "sessions.json"
        session_store.upsert("T1", session_id="s", cwd="/c", model="fable", ctx_tokens=450_000, path=store)
        h = ClaudeHandler(FakeSlack(), store_path=store)
        asyncio.run(h.initialize())
        assert h._thread_model == {"T1": "fable"}
        assert h._thread_ctx == {"T1": 450_000}

    def test_clean_run_records_ctx_in_memory_and_store(self, tmp_path, env, monkeypatch):
        from test_claude_handler import FakeProcess, _patch_subprocess
        usage = json.dumps({"type": "assistant", "message": {"content": [], "usage": {
            "input_tokens": 1000, "cache_read_input_tokens": 449_000, "output_tokens": 5}}}).encode() + b"\n"
        result = json.dumps({"type": "result", "result": "hi"}).encode() + b"\n"
        _patch_subprocess(monkeypatch, FakeProcess([usage, result]))
        h = ClaudeHandler(FakeSlack(), store_path=tmp_path / "sessions.json")
        assert asyncio.run(h._run_claude(["claude"], "p", cwd=None, thread_ts="T1")) == "hi"
        assert h._thread_ctx["T1"] == 450_000
        assert session_store.load(tmp_path / "sessions.json")["T1"]["ctx_tokens"] == 450_000
