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


MARK = "@@model thread fable\nSwitching."


class TestThreadSwitch:
    def test_marker_switches_and_reruns_once_on_resume(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=[MARK, "answer from fable"]), tmp_path)
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "use fable: do X"))
        assert reply == "answer from fable"                       # first reply suppressed
        assert len(h.runs) == 2
        assert model_flag(h.runs[0]["cmd"]) is None
        assert model_flag(h.runs[1]["cmd"]) == "fable"
        assert h.runs[1]["cmd"][h.runs[1]["cmd"].index("--resume") + 1] == "sid-1"
        assert h.runs[1]["prompt"].startswith("use fable: do X\n\n[Bridge: this session now runs on fable")
        assert h._thread_model["T1"] == "fable"
        assert session_store.load(tmp_path / "sessions.json")["T1"]["model"] == "fable"

    def test_new_message_reruns_with_resume_of_minted_session(self, tmp_path, env):
        h = make(tmp_path, env, replies=[MARK, "fable here"])
        assert asyncio.run(h.handle_message("C1", "T1", "use fable: do X")) == "fable here"
        sid = h._sessions["T1"]
        assert "--session-id" in h.runs[0]["cmd"] and "--resume" not in h.runs[0]["cmd"]
        assert h.runs[1]["cmd"][h.runs[1]["cmd"].index("--resume") + 1] == sid
        assert model_flag(h.runs[1]["cmd"]) == "fable"

    def test_noop_marker_single_run_reply_stripped(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread fable\nAlready on it."]), tmp_path)
        h._thread_model["T1"] = "fable"
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "Already on it."
        assert len(h.runs) == 1

    def test_default_clears_pick_and_reruns_without_flag(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread default", "back"]), tmp_path)
        h._thread_model["T1"] = "fable"
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "back to default")) == "back"
        assert "T1" not in h._thread_model
        assert session_store.load(tmp_path / "sessions.json")["T1"]["model"] is None
        assert model_flag(h.runs[1]["cmd"]) is None
        assert "runs on the default model" in h.runs[1]["prompt"]

    def test_marker_on_rerun_is_applied_but_not_rerun_again(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=[MARK, "@@model thread haiku\nok"]), tmp_path)
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "ok"
        assert len(h.runs) == 2
        assert h._thread_model["T1"] == "haiku"

    def test_retry_after_switch_uses_new_model(self, tmp_path, env):
        fail = "Sorry, I encountered an error processing your request."
        h = with_thread(make(tmp_path, env, replies=[MARK, fail, "ok"]), tmp_path)
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "ok"
        assert len(h.runs) == 3
        assert model_flag(h.runs[2]["cmd"]) == "fable" and "--resume" in h.runs[2]["cmd"]


class TestRefusals:
    def test_invalid_name_changes_nothing_and_tells_user(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread fable5\nSwitching."]), tmp_path)
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert reply.startswith("Switching.\n`fable5` isn't a model I can switch to.")
        assert len(h.runs) == 1 and "T1" not in h._thread_model

    def test_context_guard_refuses_small_window(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread haiku"]), tmp_path)
        h._thread_ctx["T1"] = 450_000
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert reply == "Can't switch to haiku: this thread's context is ~450k tokens, more than haiku's 200k window."
        assert len(h.runs) == 1 and "T1" not in h._thread_model

    def test_context_guard_allows_big_window_and_unknown_family(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread opus", "o", "@@model thread best", "b"]), tmp_path)
        h._thread_ctx["T1"] = 450_000
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "o"
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "b"
        assert len(h.runs) == 4 and h._thread_model["T1"] == "best"

    def test_channel_pick_without_projects_entry_notes_user(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel fable\nDone."]), tmp_path)  # no mapping
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert reply == "Done.\nThis channel has no projects.json entry, so it can't hold a default model."
        assert len(h.runs) == 1


class TestChannelPick:
    def test_sets_model_on_string_entry_reloads_and_reruns(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel fable", "ok"],
                             mapping={"C1": str(tmp_path)}), tmp_path)
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "make fable the default")) == "ok"
        assert json.loads(env.read_text()) == {"C1": {"path": str(tmp_path), "model": "fable"}}
        assert h._channel_id_to_project["C1"]["model"] == "fable"
        assert model_flag(h.runs[1]["cmd"]) == "fable"
        assert h._effective_model("T-new", "C1") == "fable"       # new threads start on it

    def test_default_removes_key_and_collapses_to_string(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel default", "ok"],
                             mapping={"C1": {"path": str(tmp_path), "model": "fable"}}), tmp_path)
        asyncio.run(h.handle_thread_reply("C1", "T1", "back to default"))
        assert json.loads(env.read_text()) == {"C1": str(tmp_path)}

    def test_default_keeps_dict_when_other_keys_present(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel default", "ok"],
                             mapping={"C1": {"path": str(tmp_path), "plugin_dir": "/x", "model": "fable"}}), tmp_path)
        asyncio.run(h.handle_thread_reply("C1", "T1", "back"))
        assert json.loads(env.read_text()) == {"C1": {"path": str(tmp_path), "plugin_dir": "/x"}}

    def test_thread_pick_shadows_channel_pick_so_no_rerun(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel haiku\nDone."],
                             mapping={"C1": str(tmp_path)}), tmp_path)
        h._thread_model["T1"] = "opus"
        assert asyncio.run(h.handle_thread_reply("C1", "T1", "hi")) == "Done."
        assert len(h.runs) == 1
        assert h._channel_id_to_project["C1"]["model"] == "haiku"  # still recorded for other threads

    def test_hand_edited_model_applies_after_reload(self, tmp_path, env):
        h = make(tmp_path, env, replies=[], mapping={"C1": str(tmp_path)})
        env.write_text(json.dumps({"C1": {"path": str(tmp_path), "model": "haiku"}}))
        asyncio.run(h.reload_projects())
        assert h._effective_model("T9", "C1") == "haiku"


FAIL = "Sorry, I encountered an error processing your request."


class TestSwitchThatCannotRun:
    def test_rerun_failing_twice_undoes_thread_pick_and_keeps_session(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model thread claude-opus-9-9", FAIL, FAIL]), tmp_path)
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert len(h.runs) == 3                                   # no further retry / scrape
        assert [model_flag(r["cmd"]) for r in h.runs[1:]] == ["claude-opus-9-9"] * 2
        assert reply == FAIL + "\nCouldn't run on claude-opus-9-9, so the switch was undone; " \
                               "this thread stays on the default model."
        assert "T1" not in h._thread_model
        assert session_store.load(tmp_path / "sessions.json")["T1"]["model"] is None
        assert h._sessions["T1"] == "sid-1"

    def test_rerun_failing_twice_undoes_channel_pick(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel claude-x-1", FAIL, FAIL],
                             mapping={"C1": str(tmp_path)}), tmp_path)
        asyncio.run(h.handle_thread_reply("C1", "T1", "hi"))
        assert json.loads(env.read_text()) == {"C1": str(tmp_path)}
        assert h._channel_id_to_project["C1"]["model"] is None


class TestChannelPickVsBigThread:
    def test_big_thread_does_not_inherit_a_too_small_channel_pick(self, tmp_path, env):
        h = make(tmp_path, env, replies=[], mapping={"C1": {"path": str(tmp_path), "model": "haiku"}})
        h._thread_ctx["T1"] = 450_000
        assert h._effective_model("T1", "C1") is None
        assert h._effective_model("T2", "C1") == "haiku"

    def test_channel_pick_from_big_thread_is_recorded_with_note(self, tmp_path, env):
        h = with_thread(make(tmp_path, env, replies=["@@model channel haiku\nDone."],
                             mapping={"C1": str(tmp_path)}), tmp_path)
        h._thread_ctx["T1"] = 450_000
        reply = asyncio.run(h.handle_thread_reply("C1", "T1", "make haiku the default"))
        assert reply == "Done.\nhaiku is now this channel's default, but this thread stays put: " \
                        "its context is ~450k tokens, more than haiku's 200k window."
        assert json.loads(env.read_text()) == {"C1": {"path": str(tmp_path), "model": "haiku"}}
        assert len(h.runs) == 1
