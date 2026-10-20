"""Unit tests for live-progress rendering (tracker, reporter, duration fmt).

These cover the pure, deterministic pieces: the tracker folds claude stream
events into a snapshot with zero clock/IO of its own, and the daemon reporter
posts/edits/deletes exactly one Slack status message. No subprocess involved.
"""

import asyncio

import pytest

import claude_handler
from claude_handler import (
    _DEFAULT_CONTEXT, _fmt_duration, _model_context, _Progress, _ProgressTracker,
)
from slack_daemon import _ProgressReporter, _reply_footer_blocks


# --- helpers to build CLI stream-json events -------------------------------

def _assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(blocks)}}


def _tool(name: str, **inp) -> dict:
    return {"type": "tool_use", "name": name, "input": inp}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _thinking(t: str) -> dict:
    return {"type": "thinking", "thinking": t}


def _tool_id(name: str, tool_id: str, **inp) -> dict:
    return {"type": "tool_use", "name": name, "id": tool_id, "input": inp}


def _tool_result(tool_use_id: str, *, is_error: bool = False, content: str = "ok") -> dict:
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_use_id,
         "is_error": is_error, "content": content}]}}


# --- _fmt_duration ---------------------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (5, "5s"), (59, "59s"),
    (60, "1m00s"), (74, "1m14s"), (599, "9m59s"),
    (3600, "1h00m"), (3661, "1h01m"),
])
def test_fmt_duration(seconds, expected):
    assert _fmt_duration(seconds) == expected


# --- _ProgressTracker ------------------------------------------------------

def test_tracker_edit_action_in_live_files_block_breakdown_in_meta():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/proj/src/session.py")))
    snap = t.snapshot(now=5.0)
    assert "✏️ Editing `session.py`" in snap.live   # action line
    assert "1 file" in snap.files and "session.py" in snap.files  # own files section
    assert "1 edit" in snap.meta                     # per-tool breakdown in muted line
    assert "5s" in snap.meta


def test_tracker_counts_unique_files_only():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/x.py")))
    t.ingest(_assistant(_tool("Write", file_path="/a/y.py")))
    t.ingest(_assistant(_tool("Edit", file_path="/a/x.py")))  # repeat
    snap = t.snapshot(now=1.0)
    assert "2 files" in snap.files   # x.py counted once, in the files section
    assert "2 edit" in snap.meta and "1 write" in snap.meta  # per-tool breakdown


def test_tracker_files_block_shows_per_file_churn():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/x.py",
                              old_string="a\nb", new_string="a\nB\nC")))
    t.ingest(_assistant(_tool("Write", file_path="/a/y.py", content="one\ntwo\nthree")))
    files = t.snapshot(now=1.0).files
    assert "📝 *2 files changed*" in files
    assert "`x.py`  +3/−2" in files   # per-file churn (Edit)
    assert "`y.py`  +3/−0" in files   # Write counts content lines as additions


def test_repeat_edits_merge_to_one_line_with_count_churn_and_monotone_numbers():
    # Editing one file N times shows ONE ticker line (`foo.py` ×N (+a/−r)) that
    # rides newest — but every re-edit still counts as a real action: numbering
    # stays monotone and "… and N more" absorbs the collapsed repeats.
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/foo.py",
                              old_string="a\nb", new_string="a\nB\nC")))   # +3/−2
    t.ingest(_assistant(_tool("Edit", file_path="/a/bar.py",
                              old_string="x", new_string="y")))            # +1/−1
    snap1 = t.snapshot(now=1.0)
    n1 = int(snap1.live.splitlines()[0].split("`")[1].rstrip("."))         # newest number
    bar1 = [l for l in snap1.live.splitlines() if "bar.py" in l][0]        # "`2.` ✏️ Editing `bar.py`"
    t.ingest(_assistant(_tool("Edit", file_path="/a/foo.py",
                              old_string="c", new_string="d")))            # +1/−1
    mid = t.snapshot(now=1.5).live.splitlines()         # snapshot BETWEEN the repeat ticks
    t.ingest(_assistant(_tool("Edit", file_path="/a/foo.py",
                              old_string="e", new_string="f")))            # +1/−1
    lines = t.snapshot(now=2.0).live.splitlines()
    foo = [l for l in lines if "foo.py" in l]
    assert len(foo) == 1                                # merged: ONE line for foo.py
    assert "×3" in foo[0] and "(+5/−4)" in foo[0]       # count + accumulated churn
    assert lines[0] == foo[0]                           # merged line rides newest
    assert lines[0].startswith("`4.`")                  # 4 real actions, number kept
    bar = [l for l in lines if "bar.py" in l]
    assert len(bar) == 1 and "×" not in bar[0]          # other file stays separate, untouched
    # THE critical bit: removing foo's stale line mid-deque must not renumber
    # anything else — bar keeps its EXACT line (number included) on every tick.
    assert bar1 in mid and bar[0] == bar1
    assert "… and 2 more" in lines[-1]                  # collapsed repeats counted honestly
    n2 = int(lines[0].split("`")[1].rstrip("."))
    assert n2 > n1                                      # numbering monotone across snapshots


def test_same_basename_edits_never_merge_across_paths():
    # a/x.py and b/x.py both render "x.py" — value-based line removal must not
    # merge or drop the WRONG file's line. State is keyed by full path and the
    # second file's display name is disambiguated (b/x.py), so lines stay unique.
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/x.py")))
    t.ingest(_assistant(_tool("Edit", file_path="/b/x.py")))
    t.ingest(_assistant(_tool("Edit", file_path="/b/x.py")))   # repeat of the SECOND file
    lines = t.snapshot(now=1.0).live.splitlines()
    xs = [l for l in lines if "x.py" in l]
    assert len(xs) == 2                                 # both files still present
    assert any("b/x.py" in l and "×2" in l for l in xs) # repeat merged on the right file
    assert any("×" not in l for l in xs)                # /a/x.py's line untouched


def test_actions_show_rolling_history_newest_first():
    # All actions so far are shown (under the cap), newest first — not just the
    # current update window.
    t = _ProgressTracker(start=0.0)
    for f in ("one", "two", "three", "four", "five"):
        t.ingest(_assistant(_tool("Read", file_path=f"/a/{f}.py")))
    lines = t.snapshot(now=1.0).live.splitlines()
    assert len(lines) == 5
    assert "five.py" in lines[0]    # newest on top
    assert "one.py" in lines[-1]    # oldest at the bottom


def test_actions_persist_across_snapshots():
    # A later snapshot still shows the earlier actions (rolling history),
    # not just what happened since the previous snapshot.
    t = _ProgressTracker(start=0.0)
    for f in ("one", "two", "three", "four", "five"):
        t.ingest(_assistant(_tool("Read", file_path=f"/a/{f}.py")))
    t.snapshot(now=1.0)  # first update
    t.ingest(_assistant(_tool("Read", file_path="/a/six.py")))
    lines = t.snapshot(now=2.0).live.splitlines()
    assert len(lines) == 6           # all six, not just the one new action
    assert "six.py" in lines[0]      # newest on top
    assert "one.py" in lines[-1]


def test_actions_are_numbered_highest_on_top():
    t = _ProgressTracker(start=0.0)
    for f in ("a", "b", "c"):
        t.ingest(_assistant(_tool("Read", file_path=f"/a/{f}.py")))
    lines = t.snapshot(now=1.0).live.splitlines()
    assert lines[0].startswith("`3.`") and "c.py" in lines[0]   # newest = highest number
    assert lines[1].startswith("`2.`")
    assert lines[2].startswith("`1.`") and "a.py" in lines[2]


def test_long_narration_is_clipped_with_ellipsis():
    t = _ProgressTracker(start=0.0)
    long = "Now it's clear and I owe you a correction so " * 12  # > 400 chars
    t.ingest(_assistant(_text(long)))
    line = t.snapshot(now=1.0).live.splitlines()[0]
    assert "💬 " in line             # numbered prefix then the narration
    assert line.endswith("…")        # clearly truncated, not cut mid-word silently
    assert not line.endswith(" …")   # trimmed on a word boundary


def test_tracker_tracks_line_churn_and_errors():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/x.py",
                              old_string="a\nb\nc", new_string="a\nB")))
    t.ingest({"type": "user", "message": {"content": [
        {"type": "tool_result", "is_error": True, "content": "boom"}]}})
    snap = t.snapshot(now=1.0)
    assert "+2/−3" in snap.files         # churn shown in the files section
    assert "1 error" in snap.meta         # COUNT only in the dimmed block
    assert "boom" not in snap.meta        # error text stays out of the dimmed block
    assert "boom" in snap.errors          # text lives in the live error block
    assert "+2/−3" in snap.summary       # also surfaced in the final summary


def test_tracker_error_block_strips_tool_use_error_tags():
    # tool_result errors arrive wrapped in <tool_use_error>…</tool_use_error>; the
    # angle brackets render as broken mrkdwn, so they must be stripped. The detail
    # goes to the live error block; the dimmed meta keeps only the count.
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "user", "message": {"content": [{"type": "tool_result",
        "is_error": True,
        "content": "<tool_use_error>File has not been read yet. Read it first.</tool_use_error>"}]}})
    snap = t.snapshot(now=1.0)
    assert "1 error" in snap.meta and "File has not been read" not in snap.meta
    assert "File has not been read yet. Read it first." in snap.errors
    assert "tool_use_error" not in snap.errors and "<" not in snap.errors


def test_tracker_tokens_turns_model_and_ctx():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "system", "subtype": "init", "model": "claude-opus-4-8"})
    usage = {"input_tokens": 1000, "output_tokens": 500,
             "cache_read_input_tokens": 99000, "cache_creation_input_tokens": 0}
    t.ingest({"type": "assistant", "message": {"usage": usage, "content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/a/x.py"}}]}})
    meta = t.snapshot(now=1.0).meta
    assert "🪙 ↑1k ↓500 ⚡99k" in meta       # in / out / cache tokens
    assert "ctx 100k/1M (10%)" in meta        # used/window AND percentage
    assert "1 turn" in meta
    assert "opus-4-8" in meta                 # model, claude- stripped


def test_tracker_ctx_uses_per_model_window():
    # Haiku 4.5 is 200K, not 1M — same prompt size, different window/%.
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-haiku-4-5", "usage": {
        "input_tokens": 100_000, "output_tokens": 1}, "content": []}})
    assert "ctx 100k/200k (50%)" in t.snapshot(now=1.0).meta


def test_model_context_loaded_from_config_with_fallback():
    assert _model_context("claude-opus-4-8") == 1_000_000   # 1M families
    assert _model_context("claude-sonnet-4-6") == 1_000_000
    assert _model_context("claude-haiku-4-5") == 200_000    # 200K
    assert _model_context("some-unknown-model") == _DEFAULT_CONTEXT


def test_model_context_falls_back_when_config_missing(monkeypatch, tmp_path):
    # Point the loader at a non-existent file → built-in windows still work.
    monkeypatch.setattr(claude_handler, "_MODEL_CONTEXT_CONFIG", tmp_path / "nope.json")
    windows, default = claude_handler._load_model_context()
    assert windows["opus"] == 1_000_000 and default == 200_000


def test_tracker_synthetic_usage_does_not_clobber_ctx():
    # A real turn sets ctx to 50%; a following "<synthetic>" mini-turn (≈0 prompt)
    # must NOT drop ctx_last to 0% — that was the live "% jumping to 5%" bug.
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 1000, "output_tokens": 10,
        "cache_read_input_tokens": 499000}, "content": []}})          # 500k/1M = 50%
    assert "ctx 500k/1M (50%)" in t.snapshot(now=1.0).meta
    t.ingest({"type": "assistant", "message": {"model": "<synthetic>", "usage": {
        "input_tokens": 2, "output_tokens": 5,
        "cache_read_input_tokens": 0}, "content": []}})
    meta = t.snapshot(now=2.0).meta
    assert "(50%)" in meta and "(0%)" not in meta                     # unchanged by synthetic


def _usage_call(ctx):  # a real assistant turn whose prompt is exactly *ctx* tokens
    return {"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 1000, "output_tokens": 10,
        "cache_read_input_tokens": ctx - 1000}, "content": []}}


def test_tracker_microcompaction_trim_inferred_from_ctx_drop():
    # No compact_boundary event, but the live prompt shrinks a lot between real
    # calls -> microcompaction (silent tool-output eviction). Show a "trimmed" tag.
    t = _ProgressTracker(start=0.0)
    t.ingest(_usage_call(200_000))
    t.ingest(_usage_call(80_000))          # 120k drop
    meta = t.snapshot(now=1.0).meta
    assert "♻️ trimmed 1×" in meta
    assert "compacted" not in meta         # not a summary compaction


def test_tracker_context_growth_is_not_a_trim():
    t = _ProgressTracker(start=0.0)
    for ctx in (50_000, 120_000, 121_000):  # monotonic growth
        t.ingest(_usage_call(ctx))
    assert "trimmed" not in t.snapshot(now=1.0).meta


def test_tracker_compaction_drop_not_double_counted_as_trim():
    # A real compact_boundary fires, THEN the next prompt drops. That drop belongs
    # to the compaction already counted — it must NOT also register as a trim.
    t = _ProgressTracker(start=0.0)
    t.ingest(_usage_call(200_000))
    t.ingest({"type": "system", "subtype": "compact_boundary"})
    t.ingest(_usage_call(50_000))
    meta = t.snapshot(now=1.0).meta
    assert "🗜️ compacted 1×" in meta and "trimmed" not in meta


def test_tracker_model_from_assistant_skips_synthetic():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "<synthetic>", "content": []}})
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "content": []}})
    assert "opus-4-8" in t.snapshot(now=1.0).meta   # real id wins; <synthetic> ignored


def test_tracker_tracks_skills_and_mcp_servers():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Skill", skill="superpowers:brainstorming")))
    t.ingest(_assistant(_tool("mcp__plugin_playwright_playwright__browser_navigate", url="x")))
    t.ingest(_assistant(_tool("mcp__claude_ai_GoDaddy__domains_suggest", query="y")))
    meta = t.snapshot(now=1.0).meta
    assert "🧩 brainstorming" in meta                 # skill, plugin prefix stripped
    assert "🔌" in meta and "plugin_playwright_playwright" in meta and "claude_ai_GoDaddy" in meta
    assert "2 mcp" in meta                            # both mcp tools grouped in the breakdown


def test_actions_cap_and_count_the_rest():
    t = _ProgressTracker(start=0.0)
    for i in range(130):
        t.ingest(_assistant(_tool("Read", file_path=f"/a/f{i}.py")))
    lines = t.snapshot(now=1.0).live.splitlines()
    assert lines[-1].startswith("… and ")           # overflow line at the bottom
    assert "f129.py" in lines[0]                     # newest on top
    assert lines[0].startswith("`130.`")            # numbered: highest on top
    shown = len(lines) - 1                           # excl. overflow line
    assert shown <= 99                               # bounded by the count cap
    hidden = int(lines[-1].split("… and ")[1].split(" ")[0])
    assert shown + hidden == 130                     # nothing lost in the count


def test_tracker_counts_compactions():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "system", "subtype": "compact_boundary"})
    t.ingest(_assistant({"type": "compaction"}))  # API compaction content block
    assert "🗜️ compacted 2×" in t.snapshot(now=1.0).meta


def test_tracker_cost_estimate_format():
    # ≥ $1 → one decimal; < $1 → two decimals.
    big = _ProgressTracker(start=0.0)
    big.ingest({"type": "assistant", "message": {"model": "claude-opus-5-5", "usage": {
        "input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 6_700_000}, "content": []}})
    assert "($2.7)" in big.snapshot(now=1.0).meta

    small = _ProgressTracker(start=0.0)
    small.ingest({"type": "assistant", "message": {"model": "claude-opus-5-5", "usage": {
        "input_tokens": 12000, "output_tokens": 4200, "cache_read_input_tokens": 368000}, "content": []}})
    assert "($0.28)" in small.snapshot(now=1.0).meta


_ONE_RUN = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 6_700_000}


def _cost_run(prior=0.0):
    t = _ProgressTracker(start=0.0, prior_cost=prior)
    t.ingest({"type": "assistant", "message": {
        "model": "claude-opus-5-5", "usage": dict(_ONE_RUN), "content": []}})
    return t


def test_tracker_cost_shows_msg_and_thread_total():
    # First message in a thread: no prior spend -> a single figure, no "total".
    m = _cost_run().snapshot(now=1.0).meta
    assert "($2.7)" in m and "total" not in m
    # A later message surfaces both this message's cost and the running thread total.
    m2 = _cost_run(prior=6.0).snapshot(now=1.0).meta
    assert "$2.7 msg" in m2 and "$8.7 total" in m2   # 6.0 + 2.69… -> $8.7


def test_tracker_thread_total_sums_unrounded_not_rounded_dollars():
    # The rounding trap: two $3.44 runs added as ROUNDED dollars ($3.4 + $3.4)
    # would read $6.8, but the true unrounded sum ($6.88) rounds to $6.9.
    c = _cost_run().run_cost()
    assert c is not None
    meta = _cost_run(prior=c).snapshot(now=1.0).meta
    assert f"{claude_handler._fmt_cost(c + c)} total" in meta        # summed, then rounded once
    assert claude_handler._fmt_cost(3.44) == "$3.4"                  # demonstrates the trap:
    assert claude_handler._fmt_cost(3.44 + 3.44) == "$6.9"          # $6.9, not $3.4+$3.4=$6.8


def test_tracker_idle_hint_at_top_when_quiet():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Bash", command="npm run build")))
    live = t.snapshot(now=60.0, idle=46.0).live
    assert live.splitlines()[0].startswith("⏳ idle ")
    # No hint when active (idle below threshold).
    t.ingest(_assistant(_tool("Read", file_path="/a/x.py")))
    assert "⏳ idle" not in t.snapshot(now=61.0, idle=2.0).live


def test_tracker_tool_breakdown_and_subagents():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Read", file_path="/a/a.py")))
    t.ingest(_assistant(_tool("Read", file_path="/a/b.py")))
    t.ingest(_assistant(_tool("Task", description="sub")))
    meta = t.snapshot(now=1.0).meta
    assert "2 read" in meta            # per-tool breakdown, busiest first
    assert "🤖 1 subagent" in meta


def test_tracker_todos_render_as_checklist():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("TodoWrite", todos=[
        {"content": "First task", "status": "completed"},
        {"content": "Second task", "status": "in_progress"},
        {"content": "Third task", "status": "pending"},
    ])))
    todos = t.snapshot(now=1.0).todos
    assert "📋 *Todos 1/3*" in todos
    assert "✅ First task" in todos
    assert "🔄 Second task" in todos
    assert "⬜ Third task" in todos


def test_tracker_todos_numbered_first_on_top():
    # Rendered in execution order, numbered from the top — the item worked first
    # reads as 1. (no reversal). Regression for the "lower number on top" ask.
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("TodoWrite", todos=[
        {"content": "Alpha", "status": "in_progress"},
        {"content": "Beta", "status": "pending"},
        {"content": "Gamma", "status": "pending"},
    ])))
    lines = t.snapshot(now=1.0).todos.splitlines()
    assert lines[1].startswith("`1.`") and "Alpha" in lines[1]
    assert lines[2].startswith("`2.`") and "Beta" in lines[2]
    assert lines[3].startswith("`3.`") and "Gamma" in lines[3]


def test_tracker_subagents_block_shows_purpose_and_status():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool_id("Task", "tu1",
                                 subagent_type="code-reviewer", description="review the diff")))
    t.ingest(_assistant(_tool_id("Task", "tu2",
                                 subagent_type="general-purpose", description="find callers")))
    snap = t.snapshot(now=1.0)
    assert "🤖 *Subagents 0/2*" in snap.subagents
    assert "🔄 code-reviewer — review the diff" in snap.subagents
    assert "🔄 general-purpose — find callers" in snap.subagents
    # purpose also rides on the action line
    assert "🤖 Subagent `code-reviewer: review the diff`" in snap.live


def test_tracker_subagent_status_flips_done_on_result():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool_id("Task", "tu1", subagent_type="x", description="do x")))
    t.ingest(_tool_result("tu1"))                      # its result arrives → done
    block = t.snapshot(now=1.0).subagents
    assert "🤖 *Subagents 1/1*" in block
    assert "✅ x — do x" in block


def test_tracker_subagent_error_result_marks_warning():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool_id("Task", "tu9", subagent_type="y", description="do y")))
    t.ingest(_tool_result("tu9", is_error=True, content="boom"))
    assert "⚠️ y — do y" in t.snapshot(now=1.0).subagents


def test_tracker_agent_tool_counts_as_subagent():
    # This environment launches subagents via the "Agent" tool, not "Task" —
    # both must populate the subagents block. Regression for an empty block.
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool_id("Agent", "ag1", subagent_type="Explore", description="map the repo")))
    snap = t.snapshot(now=1.0)
    assert "🤖 *Subagents 0/1*" in snap.subagents
    assert "🔄 Explore — map the repo" in snap.subagents
    assert "🤖 1 subagent" in snap.meta
    t.ingest(_tool_result("ag1"))
    assert "✅ Explore — map the repo" in t.snapshot(now=2.0).subagents


def test_tracker_no_subagents_block_without_tasks():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Read", file_path="/a/x.py")))
    assert t.snapshot(now=1.0).subagents == ""


def test_tracker_has_content_gates_first_post():
    # Model/usage known but nothing done yet -> no content (don't post an empty
    # "Working… 0 tools" panel). First real action flips it on.
    t = _ProgressTracker(start=0.0)
    assert t.has_content() is False
    t.ingest({"type": "system", "subtype": "init", "model": "claude-opus-4-8"})
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8",
              "usage": {"input_tokens": 9, "output_tokens": 1}, "content": []}})
    assert t.has_content() is False           # meta only, nothing to show
    t.ingest(_assistant(_tool("Read", file_path="/a/x.py")))
    assert t.has_content() is True


def test_ctx_icon_bands():
    # Independent of CTX_WARN_PCT. 🟢 0–40 · 🟡 41–50 · 🟠 51–60 · 🔴 61–80 · 🚨 81+
    assert claude_handler._ctx_icon(40) == "🟢"
    assert claude_handler._ctx_icon(41) == "🟡"
    assert claude_handler._ctx_icon(50) == "🟡"
    assert claude_handler._ctx_icon(51) == "🟠"
    assert claude_handler._ctx_icon(60) == "🟠"
    assert claude_handler._ctx_icon(61) == "🔴"
    assert claude_handler._ctx_icon(80) == "🔴"
    assert claude_handler._ctx_icon(81) == "🚨"


def test_meta_ctx_readout_carries_health_icon():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 450_000, "output_tokens": 1}, "content": []}})  # 45% -> yellow
    assert "🟡 ctx 450k/1M (45%)" in t.snapshot(now=1.0).meta


def test_tracker_ctx_warning_at_and_above_threshold():
    # 55% >= default CTX_WARN_PCT (51) -> nudge suffix on the ctx readout in the
    # META (so it rides on the reply, which no longer shows the live actions view).
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 550_000, "output_tokens": 1}, "content": []}})  # 550k/1M = 55%
    meta = t.snapshot(now=1.0).meta
    assert "🟠 ctx 550k/1M (55%) — consider a new session" in meta


def test_tracker_no_ctx_warning_below_threshold(monkeypatch):
    monkeypatch.setattr(claude_handler, "CTX_WARN_PCT", 45)
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 300_000, "output_tokens": 1}, "content": []}})  # 30%
    assert "consider a new session" not in t.snapshot(now=1.0).meta


def test_tracker_ctx_warning_disabled_when_zero(monkeypatch):
    monkeypatch.setattr(claude_handler, "CTX_WARN_PCT", 0)
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "assistant", "message": {"model": "claude-opus-4-8", "usage": {
        "input_tokens": 900_000, "output_tokens": 1}, "content": []}})  # 90%
    assert "consider a new session" not in t.snapshot(now=1.0).meta


def test_tracker_dirty_lifecycle():
    t = _ProgressTracker(start=0.0)
    assert t.dirty is False                       # nothing yet
    t.ingest(_assistant(_tool("Read", file_path="/a/b.py")))
    assert t.dirty is True                        # event arrived
    t.snapshot(now=1.0)
    assert t.dirty is False                       # consumed by snapshot
    t.ingest({"type": "system", "subtype": "init"})  # non-activity event
    assert t.dirty is False


def test_tracker_text_and_thinking_update_action():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_thinking("let me look at this")))
    assert "🤔 Thinking" in t.snapshot(now=1.0).live
    t.ingest(_assistant(_text("Now I'll refactor the handler\nsecond line")))
    live = t.snapshot(now=2.0).live
    assert "💬 Now I'll refactor the handler" in live
    assert "second line" not in live  # only the first line is shown


def test_tracker_summary_is_one_line_headline_detail_kept_in_sections():
    t = _ProgressTracker(start=0.0)
    t.ingest(_assistant(_tool("Edit", file_path="/a/session.py")))
    t.ingest(_assistant(_tool("Read", file_path="/a/config.py")))
    t.ingest(_assistant(_tool("Bash", command="pytest -q\necho done")))
    snap = t.snapshot(now=134.0, done=True)
    assert snap.done is True
    assert snap.summary == "✅ Done · 1 file changed · 3 tools · 2m14s"  # one-line headline
    assert "```" not in snap.summary          # no collapsed detail block now
    # Detail lives in the kept sections instead:
    assert "session.py" in snap.files
    assert "1 read · 1 edit · 1 bash" in snap.meta or "1 bash" in snap.meta


def test_tracker_summary_no_detail_block_when_nothing_tracked():
    t = _ProgressTracker(start=0.0)
    t.ingest({"type": "system", "subtype": "init"})  # nothing trackable
    summary = t.snapshot(now=3.0, done=True).summary
    assert summary == "✅ Done · 0 tools · 3s"  # headline only, no fence
    assert "```" not in summary


# --- _ProgressReporter (daemon side) ---------------------------------------

class _FakeClient:
    def __init__(self):
        self.posts: list[dict] = []
        self.updates: list[dict] = []
        self.deletes: list[dict] = []

    async def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)
        return {"ts": "1700000000.0001"}

    async def chat_update(self, **kwargs):
        self.updates.append(kwargs)

    async def chat_delete(self, **kwargs):
        self.deletes.append(kwargs)


def _has_actions(blocks):
    return any(b.get("type") == "actions" for b in blocks)


def test_reporter_creates_then_edits_one_message():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")

    asyncio.run(reporter(_Progress(live="🔄 step 1", summary="s", done=False)))
    asyncio.run(reporter(_Progress(live="🔄 step 2", summary="s", done=False)))

    assert len(client.posts) == 1                       # only ONE new message
    assert client.posts[0]["text"] == "🔄 step 1"
    assert client.posts[0]["thread_ts"] == "T1"
    assert len(client.updates) == 1                      # second was an in-place edit
    assert client.updates[0]["text"] == "🔄 step 2"
    assert client.updates[0]["ts"] == "1700000000.0001"


def test_reporter_live_message_has_no_button():
    # Stop is a 🛑 reaction the daemon adds, not an inline button.
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(live="🔄 working", summary="s", done=False)))
    assert not _has_actions(client.posts[0]["blocks"])


def test_reporter_calls_on_status_posted_with_ts_once():
    client = _FakeClient()
    hits = []

    async def hook(status_ts):
        hits.append(status_ts)

    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1", on_status_posted=hook)
    asyncio.run(reporter(_Progress(live="x", summary="s", done=False)))
    asyncio.run(reporter(_Progress(live="y", summary="s", done=False)))
    assert hits == ["1700000000.0001"]          # fired once, with the message ts
    assert reporter.posted_ts == "1700000000.0001"


def test_reporter_renders_meta_as_context_block():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(
        live="✏️ Editing `x.py`", summary="s", done=False, meta="🔄 1 file · 2 tools · 5s",
    )))
    blocks = client.posts[0]["blocks"]
    kinds = [b["type"] for b in blocks]
    # rich_text (banner + Actions title); rich_text (action log, never collapses); context (tally).
    assert kinds == ["rich_text", "rich_text", "context"]
    assert "not the reply" in _rich_text(blocks[0])         # banner folded into the header
    assert blocks[-1]["elements"][0]["text"] == "🔄 1 file · 2 tools · 5s"


def _rich_text(block: dict) -> str:
    """Flatten a rich_text block's runs back to plain text (for assertions)."""
    return "".join(
        "".join(r.get("text", "") for r in sec["elements"])
        for sec in block["elements"]
    )


def test_reporter_renders_todos_as_rich_text_never_collapsing():
    # Todos are a rich_text block, not a section: Slack never gives rich_text a
    # "Show more" collapse, so the checklist always shows in full at normal size.
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(
        live="🔄 working", summary="s", done=False,
        meta="2 read · 5s", todos="📋 *Todos 1/2*\n`1.` ✅ a\n`2.` ⬜ b",
    )))
    blocks = client.posts[0]["blocks"]
    kinds = [b["type"] for b in blocks]
    assert kinds == ["rich_text", "rich_text", "context"]  # banner+todos, actions, tally
    assert "not the reply" in _rich_text(blocks[0])       # banner folded into the header
    assert "Todos 1/2" in _rich_text(blocks[0])
    assert "🔄 working" in _rich_text(blocks[1])           # actions stay their own block
    # Inline styles survive the conversion: the bold header and the `1.` code prefix.
    runs = blocks[0]["elements"][0]["elements"]
    assert any(r.get("style", {}).get("bold") for r in runs)
    assert any(r.get("style", {}).get("code") for r in runs)


def test_reporter_combines_todos_and_subagents_in_one_rich_text_with_blank_line():
    # Adjacent blocks render flush, so todos + subagents go in ONE rich_text block
    # with a blank line between them (the visible gap). Still never collapses.
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(
        live="🔄 working", summary="s", done=False, meta="2 tools · 5s",
        todos="📋 *Todos 1/2*\n`1.` ✅ a\n`2.` ⬜ b",
        subagents="🤖 *Subagents 0/1*\n🔄 reviewer — check",
    )))
    blocks = client.posts[0]["blocks"]
    kinds = [b["type"] for b in blocks]
    assert kinds == ["rich_text", "rich_text", "context"]   # banner+todos+subagents, actions, tally
    flat = _rich_text(blocks[0])
    assert "not the reply" in flat                         # banner folded into the header
    assert "Todos 1/2" in flat and "Subagents 0/1" in flat
    assert flat.index("Todos") < flat.index("Subagents")  # todos lead
    assert "\n\n" in flat                                  # blank line separates them
    assert "🔄 working" in _rich_text(blocks[1])


def test_reporter_fast_run_keeps_meta_without_posting_status():
    # A fast run finishes before any live status was posted: the done snapshot
    # must capture meta (for the reply line) but NOT spawn a new status message.
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(live="x", summary="✅ Done", done=True, meta="2 read · 3s")))
    assert client.posts == [] and client.updates == []   # stayed silent
    assert reporter.last_meta == "2 read · 3s"            # meta still captured


def test_reporter_remembers_last_meta_for_reply():
    # The reporter keeps the latest meta so the daemon can reuse the same line
    # under the final reply (DRY with the live status block).
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(live="x", summary="s", done=False, meta="2 read · 5s")))
    asyncio.run(reporter(_Progress(live="y", summary="s", done=True, meta="3 read · 9s")))
    assert reporter.last_meta == "3 read · 9s"


def test_errors_block_is_live_only_not_in_reply_footer():
    prog = _Progress(
        live="🔄 working", summary="s", done=True,
        meta="3 tools · 1 error · 5s",                 # dimmed: count only
        errors="*1 error*\n• boom",                    # detail block (live only)
        files="📝 *1 file changed*\n• `x.py`",
    )
    reporter = _ProgressReporter(_FakeClient(), channel="C1", thread_ts="T1")
    live = reporter._live_blocks(prog)
    assert any("• boom" in b["text"]["text"] for b in live if b["type"] == "section")
    # reply footer: files + meta (count) only — no error DETAIL block
    reporter.last_progress = prog
    footer = _reply_footer_blocks(reporter)
    assert not any(b["type"] == "section" and "• boom" in b["text"]["text"] for b in footer)
    assert any("file changed" in b["text"]["text"] for b in footer if b["type"] == "section")
    assert "1 error" in footer[-1]["elements"][0]["text"]  # count rides on the meta


def test_reporter_renders_subagents_as_rich_text():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(
        live="🔄 working", summary="s", done=False,
        meta="2 tools · 5s", subagents="🤖 *Subagents 0/1*\n🔄 reviewer — check",
    )))
    blocks = client.posts[0]["blocks"]
    kinds = [b["type"] for b in blocks]
    assert kinds == ["rich_text", "rich_text", "context"]  # banner+subagents, actions, tally
    assert "Subagents 0/1" in _rich_text(blocks[0])


def test_reporter_renders_files_as_its_own_section():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(
        live="🔄 working", summary="s", done=False,
        meta="1 edit · 5s", files="📝 *1 file changed*\n• `x.py`  +3/−1",
    )))
    blocks = client.posts[0]["blocks"]
    kinds = [b["type"] for b in blocks]
    # rich_text (banner + Actions title); rich_text (actions log); section (files); context (tally).
    assert kinds == ["rich_text", "rich_text", "section", "context"]
    assert "1 file changed" in blocks[2]["text"]["text"]


def test_reporter_done_snapshot_renders_summary():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(live="🔄 working", summary="✅ done", done=False)))
    asyncio.run(reporter(_Progress(live="🔄 working", summary="✅ done", done=True)))
    assert client.updates[-1]["text"] == "✅ done"       # summary, not live
    # Even a lingering done window must read as "not the reply" (delete() can lag);
    # the summary rides bold in the rich_text header, so the blank-line gap applies.
    blocks = client.updates[-1]["blocks"]
    assert blocks[0]["type"] == "rich_text"
    assert "not the reply" in _rich_text(blocks[0])
    assert not _has_actions(blocks)
    assert reporter.posted_ts == "1700000000.0001"  # kept for reaction cleanup


def test_reporter_delete_removes_posted_message():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter(_Progress(live="🔄 working", summary="s", done=False)))
    asyncio.run(reporter.delete())
    assert client.deletes == [{"channel": "C1", "ts": "1700000000.0001"}]


def test_reporter_delete_is_noop_when_never_posted():
    client = _FakeClient()
    reporter = _ProgressReporter(client, channel="C1", thread_ts="T1")
    asyncio.run(reporter.delete())  # short run never posted a status
    assert client.deletes == []
    assert client.posts == []
