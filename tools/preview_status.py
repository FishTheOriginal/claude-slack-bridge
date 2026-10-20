#!/usr/bin/env python3
"""Preview the live Slack status message locally — no Slack, no daemon, no restart.

Feeds a scripted sequence of claude stream-json events through the real
_ProgressTracker, renders the blocks the daemon would post via the real
_ProgressReporter, and prints a text approximation of how Slack would show it.

Run:  ./.venv/bin/python tools/preview_status.py
Use it to eyeball layout changes before merging/deploying.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claude_handler import _ProgressTracker  # noqa: E402
from slack_daemon import _ProgressReporter  # noqa: E402


def _assistant(*blocks, usage=None, model=None):
    msg = {"content": list(blocks)}
    if usage is not None:
        msg["usage"] = usage
    if model is not None:
        msg["model"] = model
    return {"type": "assistant", "message": msg}


def tool(name, **inp):
    return {"type": "tool_use", "name": name, "input": inp}


def text(t):
    return {"type": "text", "text": t}


def tool_error(msg):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "is_error": True, "content": msg}]}}


def task(tool_id, subagent_type, description):
    # a tool_use BLOCK — wrap in _assistant(...) like tool().
    return {"type": "tool_use", "name": "Task", "id": tool_id,
            "input": {"subagent_type": subagent_type, "description": description}}


def task_result(tool_use_id, *, is_error=False):
    return {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_use_id,
         "is_error": is_error, "content": "done"}]}}


# A representative run: model+usage, edits with churn, a bash command carrying a
# secret (redacted), reads, a skill, MCP tools, two subagents (one done / one
# running), numbered todos, an error — and a context-% high enough to nudge.
SCENARIO = [
    _assistant(text("Now it's clear — I'll wire the tracker into the daemon and add tests."),
               model="claude-opus-4-8"),
    {"type": "system", "subtype": "compact_boundary"},  # context auto-compacted
    _assistant(tool("Read", file_path="/proj/src/claude_handler.py")),
    _assistant(tool("Edit", file_path="/proj/src/claude_handler.py",
                    old_string="a\nb\nc\nd", new_string="a\nB")),
    _assistant(tool("Edit", file_path="/proj/src/slack_daemon.py",
                    old_string="x", new_string="x\ny\nz")),
    _assistant(tool("Write", file_path="/proj/tools/preview_status.py",
                    content="\n".join(["line"] * 40))),
    # A secret in the command line must be masked before it reaches Slack.
    _assistant(tool("Bash",
                    command="ssh node@host 'T=ogk2r42gpi01pe7vwd4ljgdmhow3ye69vjrfi91asy4n3uyj; for d in 2026-06-25 2026-06-26; do sync $d; done'")),
    # A build command that exits non-zero: the error block should name WHAT ran
    # and gloss the cryptic code (144 = killed by signal 16) — and the OUT=<path>
    # must stay readable (a path is not a secret, entropy pass leaves it alone).
    _assistant({"type": "tool_use", "name": "Bash", "id": "b7", "input": {
        "command": "OUT=/home/node/proj/meridian-dashboard/.pdfbuild; mkdir -p $OUT; md2html build"}}),
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "b7", "is_error": True, "content": "Exit code 144"}]}},
    _assistant(tool("Skill", skill="superpowers:brainstorming")),
    _assistant(tool("mcp__plugin_playwright_playwright__browser_navigate", url="http://localhost:3000")),
    _assistant(tool("mcp__claude_ai_GoDaddy__domains_suggest", query="example")),
    _assistant(task("tu1", "code-reviewer", "review the diff for bugs")),
    _assistant(task("tu2", "general-purpose", "find every caller of _meta")),
    task_result("tu1"),                                  # first subagent finishes
    tool_error("pytest: 1 failed in test_progress.py"),
    _assistant(tool("TodoWrite", todos=[
        {"content": "Add per-file churn", "status": "completed"},
        {"content": "Wire ctx used/window", "status": "in_progress"},
        {"content": "Update tests", "status": "pending"},
        {"content": "Preview + deploy", "status": "pending"},
    ])),
    _assistant(tool("Grep", pattern="def _meta")),
]


class _CaptureClient:
    def __init__(self):
        self.blocks = None

    async def chat_postMessage(self, **kw):
        self.blocks = kw.get("blocks")
        return {"ts": "1.1"}

    async def chat_update(self, **kw):
        self.blocks = kw.get("blocks")


def render(blocks) -> str:
    out = []
    for b in blocks or []:
        if b["type"] == "section":
            out.append(b["text"]["text"])
        elif b["type"] == "rich_text":
            # Flatten runs back to text (Slack renders this normal-size, no collapse).
            for sec in b["elements"]:
                out.append("".join(r.get("text", "") for r in sec["elements"]))
        elif b["type"] == "context":
            for el in b["elements"]:
                # Slack renders context blocks small/grey — prefix to suggest "muted".
                for line in el["text"].splitlines():
                    out.append(f"\x1b[90m{line}\x1b[0m")
        out.append("\x1b[90m" + "·" * 60 + "\x1b[0m")
    return "\n".join(out)


def show(title, progress):
    client = _CaptureClient()
    asyncio.run(_ProgressReporter(client, "C", "T")(progress))
    print(f"\n\x1b[1m=== {title} ===\x1b[0m")
    print(render(client.blocks))


def main():
    # Per-API-call usage: each turn re-reads a large cached prompt (cache_read
    # accumulates into a big ⚡ total) while the last call's prompt size is what
    # ctx shows. Mirrors a real long run (↑ small, ↓ small, ⚡ big).
    per_call = {"input_tokens": 150, "output_tokens": 15,
                "cache_read_input_tokens": 610000, "cache_creation_input_tokens": 0}
    # prior_cost simulates earlier messages in the thread, so the meta shows both
    # this message's cost and the running thread total ("$x msg · $y total").
    t = _ProgressTracker(start=0.0, prior_cost=6.0)
    for ev in SCENARIO:
        if ev["type"] == "assistant" and "usage" not in ev["message"]:
            ev["message"]["usage"] = dict(per_call)
        t.ingest(ev)
    show("LIVE (while running)", t.snapshot(now=134.0))
    show("LIVE (quiet — idle hint)", t.snapshot(now=180.0, idle=46.0))
    show("FINAL SUMMARY (done)", t.snapshot(now=134.0, done=True))


if __name__ == "__main__":
    main()
