"""
claude_handler.py — Spawns Claude Code CLI subprocesses for Human→Claude tasks.

When a human posts a message in Slack, this handler runs ``claude -p`` to
generate a response.  Thread continuations use ``--resume`` so Claude retains
full context (tool use, reasoning) across messages in the same thread.

If the session ID is lost (e.g. container restart), falls back to a one-shot
``claude -p`` with the formatted thread history as the prompt.

Project detection: reads ``projects.json`` at the repo root to map Slack
channels to project directories.  When a message arrives, the handler resolves
the channel to a project path and runs ``claude -p`` from that directory so
Claude sees the project's CLAUDE.md and codebase.

Each entry in ``projects.json`` can be a plain path string (legacy) or a dict
with ``path`` and optional ``plugin_dir`` / ``worktrees`` / ``model`` fields.
When ``plugin_dir`` is set, ``--plugin-dir <dir>`` is prepended to the
``claude -p`` invocation so project-specific skills are loaded automatically.
``model`` is the channel's default ``--model`` (a thread's own pick wins).

When ``worktrees`` is a ``{label: path}`` map, users can route a top-level
Slack message to a specific worktree by prefixing the message with
``[label]`` (e.g. ``@Bot [feature-x] refactor session.py``). The label
prefix is stripped before the prompt is sent to Claude. Replies inside the
resulting thread stay in that worktree without re-tagging.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import session_store
import signal
import stat
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, NamedTuple

import quota
from security import redact_secrets

logger = logging.getLogger(__name__)

# A run is killed only after IDLE_TIMEOUT seconds with NO stream activity
# (genuinely stuck), not on total wall-clock time — so long tasks run to
# completion and the user stops them with 🛑 instead of hitting a fixed cap.
# A long Bash/build emits no events while running, so keep this generous; set
# IDLE_TIMEOUT_SECONDS=0 to disable the watchdog and rely solely on 🛑.
IDLE_TIMEOUT = float(os.environ.get("IDLE_TIMEOUT_SECONDS", "1800"))  # 30 min
# Live progress: aggregate stream events and edit a single Slack status
# message at most once every PROGRESS_INTERVAL seconds (≤0 falls back to 10).
PROGRESS_INTERVAL = float(os.environ.get("PROGRESS_INTERVAL_SECONDS", "10"))
# Max recent actions the live status shows (rolling, newest first). Beyond this
# the rest collapse into "… and N more". ≤0 falls back to 99.
PROGRESS_ACTIONS = int(os.environ.get("PROGRESS_ACTIONS", "99"))
# After this many seconds with no stream event, the live status shows an "⏳ idle"
# hint (the run is on a long quiet step, e.g. a slow build). Distinct from
# IDLE_TIMEOUT, which kills the run; this only annotates it.
IDLE_HINT = float(os.environ.get("IDLE_HINT_SECONDS", "30"))
# Once a call's prompt reaches this % of the model's context window, the live
# status nudges the user to start a fresh session/thread (long contexts get
# slower, pricier and more error-prone). Default 51 = the 🟠 band start, so the
# 🟡 41–50 band is a silent heads-up and the nudge fires when it turns orange.
# Independent of the icon bands (_CTX_ICON_BANDS). Set CTX_WARN_PCT=0 to disable.
CTX_WARN_PCT = int(os.environ.get("CTX_WARN_PCT", "51"))
# When a thread's context is at/above CTX_WARN_PCT, append a one-off note to the
# next turn's prompt so the agent reminds the user (in its own voice) to wrap up
# and continue in a fresh session. Off => only the silent context-fill indicator.
CTX_NUDGE_AGENT = os.environ.get("CTX_NUDGE_AGENT", "true").strip().lower() not in (
    "0", "false", "no", "off",
)
# Appended to the turn's prompt (active turn, not system prompt — more reliably
# acted on) while ctx >= CTX_WARN_PCT. Two signals: the checkpoint clause is
# always included (the agent decides WHEN to raise it); the urgency clause is
# added only when context crosses into a worse band (once per escalation).
_CTX_NOTE_HEAD = "this session is ~{pct}% full."
_CTX_NOTE_URGENCY = (
    " It just crossed into a heavier usage zone: tell the user in one line that "
    "it's time to plan a fresh session."
)
_CTX_NOTE_CHECKPOINT = (
    " At a natural checkpoint (task done, committed, tests green), recommend a "
    "fresh thread now and offer a short handoff; mid-task, don't raise it. You "
    "can't switch sessions yourself."
)


# Context-fill indicator bands: (lower-bound %, icon), highest first. Cosmetic and
# independent of CTX_WARN_PCT (the nudge threshold) — edit here to retune.
#   🟢 0–40 · 🟡 41–50 · 🟠 51–60 · 🔴 61–80 · 🚨 81+
_CTX_ICON_BANDS: tuple[tuple[int, str], ...] = (
    (81, "🚨"), (61, "🔴"), (51, "🟠"), (41, "🟡"), (0, "🟢"),
)


def _ctx_icon(pct: int) -> str:
    """Return the context-fill indicator for *pct* (see _CTX_ICON_BANDS)."""
    for lo, icon in _CTX_ICON_BANDS:
        if pct >= lo:
            return icon
    return "🟢"


def _ctx_band(pct: int) -> int:
    """Severity band index (0 green … 4 critical), derived from _CTX_ICON_BANDS.

    Used to re-fire the agent nudge only when context crosses into a *worse* band
    (🟠→🔴→🚨), not on every reply.
    """
    return sum(1 for lo, _ in _CTX_ICON_BANDS if lo > 0 and pct >= lo)

# A progress callback receives an immutable snapshot of run progress and pushes
# it to Slack (create/edit a status message). Owned by the daemon; the handler
# only decides *when* to call it (throttled aggregation + a final summary).
ProgressCb = Callable[["_Progress"], Awaitable[None]]
# Claude CLI in stream-json mode emits one JSON event per line. A single
# event can embed large tool inputs/results (file reads, MCP responses,
# task outputs), easily exceeding asyncio's default 64 KB StreamReader
# buffer and raising ``LimitOverrunError`` ("Separator is found, but chunk
# is longer than limit"). Bump the limit so we can ingest realistic events.
STREAM_BUFFER_LIMIT = 100 * 1024 * 1024  # 100 MB
PROJECTS_CONFIG = Path(__file__).parent.parent / "projects.json"

# Allow Slack's leading bold/italic/strike markers (``*``, ``_``, ``~``)
# before the tag — Slack delivers ``*[label] msg*`` when the user bolds
# the whole line.
_WORKTREE_TAG_RE = re.compile(r"^[\s*_~]*\[([^\]]+)\]\s*")
# Labels become directory names; restrict to a safe alphabet to block
# path-traversal attempts like ``[../etc]``.
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# A ``@@model <thread|channel> <model|default>`` line Claude emits to switch the
# model; tolerates a stray pair of backticks (the model sometimes code-formats it).
_MODEL_MARKER_RE = re.compile(r"^\s*`?@@model\s+(thread|channel)\s+([^\s`]+)`?\s*$", re.IGNORECASE)
# Model names the claude CLI accepts for --model: aliases or full ids, optional [1m].
_MODEL_NAME_RE = re.compile(r"^(?:opus|sonnet|haiku|fable|best|opusplan|claude-[a-z0-9.-]+)(?:\[1m\])?$")

# A claude -p run that failed transiently (CLI missing, non-zero rc, or no
# result event) returns one of these fixed user-facing strings from
# _run_claude. We treat any of them as a resume failure rather than parsing
# stderr (brittle). A timeout is deliberately NOT here — see _TIMEOUT_REPLY.
_RUN_FAILURE_SENTINELS = frozenset({
    "Sorry, the Claude CLI is not available.",
    "Sorry, I encountered an error processing your request.",
    "Sorry, I couldn't parse the response.",
})

# A non-zero exit may append a classified reason to this stem (e.g. "… (rate
# limited — try again shortly)"). The exact stem stays in _RUN_FAILURE_SENTINELS
# so a bare failure still matches; the prefix lets the resume policy recognise the
# enriched variants too. The stem is long and specific enough that a genuine
# model reply never starts with it.
_RUN_FAILURE_PREFIX = "Sorry, I encountered an error processing your request."


def _is_run_failure(reply: str) -> bool:
    """True for any _run_claude failure reply, plain or reason-enriched."""
    return reply in _RUN_FAILURE_SENTINELS or reply.startswith(_RUN_FAILURE_PREFIX)


def _result_error_text(ev: "dict | None") -> str:
    """Flatten a `result` stream event's text fields, for error classification.

    A usage/quota limit can arrive as a result event (often with empty stderr),
    so its subtype/result/error/message text is folded into the same blob the
    stderr classifier sees. Empty string when there's nothing useful.
    """
    if not isinstance(ev, dict):
        return ""
    parts: list[str] = []
    for key in ("subtype", "result", "error", "message"):
        v = ev.get(key)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(json.dumps(v))
    return " ".join(parts)


# stderr substrings -> short human reason, checked in order (most specific first).
_ERROR_SIGNATURES: list[tuple[tuple[str, ...], str]] = [
    (("usage limit", "usage_limit", "limit reached", "limit will reset",
      "5-hour limit", "weekly limit"),
     "Claude usage limit reached — try again later or check the plan"),
    (("rate limit", "rate_limit", "429"), "rate limited — try again shortly"),
    (("overloaded", "529"), "the API is overloaded — try again shortly"),
    (("credit balance", "insufficient", "quota", "billing", "payment required", " 402"),
     "credit/quota issue — check the Anthropic account"),
    (("invalid api key", "invalid x-api-key", "authentication", "unauthorized",
      "forbidden", " 401", " 403"),
     "authentication failed — check the API key"),
    (("prompt is too long", "context length", "maximum context", "too many tokens",
      "context_length_exceeded"),
     "the context is too large — start a new session or shorten the request"),
    (("internal server error", "bad gateway", "gateway timeout", "service unavailable",
      " 500", " 502", " 503", " 504"),
     "an upstream server error — try again"),
    (("timed out", "timeout", "econnreset", "econnrefused", "getaddrinfo",
      "network", "dns"),
     "a network problem — try again"),
]


def _classify_run_error(stderr_lines: list[str]) -> "str | None":
    """Map captured stderr to a short, user-facing reason (None if unrecognised).

    On no match but non-empty stderr, returns the last line (redacted + clipped)
    so the reply still says something concrete rather than a generic apology.
    """
    blob = " ".join(stderr_lines).lower()
    if not blob.strip():
        return None
    for needles, reason in _ERROR_SIGNATURES:
        if any(n in blob for n in needles):
            return reason
    # Fall back to the last meaningful line (skip blank/placeholder entries).
    last = next((ln.strip() for ln in reversed(stderr_lines) if ln.strip()), "")
    return redact_secrets(_clip(last, 160)) if last else None

# A run terminated by a signal (returncode < 0 — e.g. an OOM kill or a
# deliberate stop) is NOT a transient failure: retrying it just re-executes work
# that was intentionally or unavoidably halted. Surface a distinct reply that the
# resume policy does NOT retry. Intentionally NOT in _RUN_FAILURE_SENTINELS.
_INTERRUPTED_REPLY = "Sorry, the run was interrupted before it finished."

# A run the idle watchdog kills (no activity for IDLE_TIMEOUT) is NOT a transient
# failure: it was either making progress and stalled, or genuinely stuck, so
# retrying (then scraping) just serially re-runs the same expensive work while
# the user sees nothing. Surface this immediately and do NOT retry. Intentionally
# NOT in _RUN_FAILURE_SENTINELS.
_TIMEOUT_REPLY = "Sorry, the run stalled with no activity and was stopped. Try again or break it into a smaller step."

# Notes appended to the posted reply when an @@model marker is refused or the
# re-run after a pick fails, and the line appended to the re-sent message once a
# pick is recorded (see _run_session).
_NOT_A_MODEL = (
    "`{name}` isn't a model I can switch to. Use opus, sonnet, haiku, fable, best or "
    "opusplan (optionally with [1m]), or a full claude-… id."
)
_CTX_TOO_BIG = (
    "Can't switch to {model}: this thread's context is ~{ctx} tokens, more than "
    "{model}'s {window} window."
)
_CTX_TOO_BIG_HERE = (
    "{model} is now this channel's default, but this thread stays put: its context is "
    "~{ctx} tokens, more than {model}'s {window} window."
)
_NO_CHANNEL_ENTRY = "This channel has no projects.json entry, so it can't hold a default model."
_PROJECTS_UNWRITABLE = (
    "Couldn't update projects.json (it isn't valid JSON or can't be written), so the "
    "channel default is unchanged."
)
_SWITCH_FAILED = "Couldn't run on {model}, so the switch was undone; this thread stays on {old}."
_RERUN_FAILED = "The model pick is saved, but the run after it failed; please send the message again."
_SWITCH_NOTE = "\n\n[Bridge: the model pick is saved; this session runs on {model}. Now answer the message above.]"

# Prepended to a reply produced by the scrape fallback, so the user knows the
# thread's session was replaced and only the Slack messages carried over.
_NEW_SESSION_NOTE = "⚠️ New session: I couldn't resume the earlier one, so I only have this thread's messages, not my earlier work.\n\n"


def _jsonl_path(cwd: str | None, session_id: str) -> Path:
    """Local JSONL path for a Claude session.

    Claude Code stores sessions at
    ``~/.claude/projects/<encoded-cwd>/<session_id>.jsonl`` where the encoded
    cwd is the absolute project path with every non-alphanumeric character
    replaced by ``-`` (e.g. ``/home/node/proj/x`` → ``-home-node-proj-x``).
    """
    abs_cwd = os.path.abspath(cwd) if cwd else os.getcwd()
    encoded = re.sub(r"[^A-Za-z0-9]", "-", abs_cwd)
    return Path.home() / ".claude" / "projects" / encoded / f"{session_id}.jsonl"


def _parse_worktree_tag(text: str) -> tuple[str | None, str]:
    """Strip a leading ``[label]`` tag from *text*.

    Returns ``(label, remaining_text)``. ``label`` is ``None`` when no tag
    is present or when the label contains unsafe characters. The label is
    what users type in Slack to route a Flow-B message to a specific
    worktree (e.g. ``[claude-slack-test] hi``).
    """
    match = _WORKTREE_TAG_RE.match(text)
    if not match:
        return None, text
    label = match.group(1).strip()
    if not _SAFE_LABEL_RE.match(label):
        return None, text
    remaining = text[match.end() :]
    return label, remaining


def _parse_model_markers(reply: str) -> tuple[str, list[tuple[str, str]]]:
    """Split *reply* into (clean_text, [(scope, name), ...]).

    ``@@model <thread|channel> <model|default>`` lines are removed and collected
    in order (lower-cased); the rest is rejoined and stripped, like ``@@attach``.
    """
    kept: list[str] = []
    picks: list[tuple[str, str]] = []
    for line in reply.splitlines():
        m = _MODEL_MARKER_RE.match(line)
        if m:
            picks.append((m.group(1).lower(), m.group(2).lower()))
        else:
            kept.append(line)
    return "\n".join(kept).strip(), picks


def _join_notes(text: str, notes: list[str]) -> str:
    """*text* followed by one line per bridge note; unchanged when there are none."""
    return "\n".join([text, *notes]).strip() if notes else text


def _resolve_dynamic_worktree(default_path: str, label: str) -> str | None:
    """Resolve *label* to a sibling worktree directory of *default_path*.

    Worktrees are typically created with ``git worktree add ../<name>`` so
    they live next to the main checkout. This lets users add/remove
    worktrees without editing ``projects.json``: the daemon checks whether
    a sibling directory named *label* exists and looks like a git checkout
    (has a ``.git`` file or directory).

    Returns the resolved path or ``None`` if no matching directory exists.
    """
    parent = os.path.dirname(default_path)
    candidate = os.path.join(parent, label)
    git_marker = os.path.join(candidate, ".git")
    if os.path.isdir(candidate) and os.path.exists(git_marker):
        return candidate
    return None


def _load_project_map() -> dict[str, Any]:
    """Load channel → project config mapping from projects.json.

    Values may be a plain path string (legacy) or a dict with ``path`` and
    optional ``plugin_dir`` keys (extended format).
    """
    if not PROJECTS_CONFIG.exists():
        logger.warning("No projects.json at %s — project detection disabled.", PROJECTS_CONFIG)
        return {}
    with open(PROJECTS_CONFIG, encoding="utf-8") as f:
        mapping = json.load(f)
    logger.info("Loaded project map with %d entries.", len(mapping))
    return mapping


def _append_attachment_note(prompt: str, files: list[tuple[str, str]]) -> str:
    """Append a ``[User attached: ...]`` note to *prompt* for downloaded files.

    *files* is the ``(local_path, mimetype)`` list from
    ``attachments.download_files``. Claude views image paths visually and
    ``Read``s document paths; only the path + mimetype are passed.
    """
    if not files:
        return prompt
    parts = ", ".join(f"{path} ({mimetype})" for path, mimetype in files)
    return f"{prompt}\n\n[User attached: {parts}]"


def _descendant_pids(pid: int) -> list[int]:
    """All descendant PIDs of *pid*, via the Linux /proc children interface.

    Walked while the tree is still intact (callers MUST snapshot before killing
    the parent — afterwards the children reparent to init and can't be traced
    from *pid*). Returns [] if the children interface is unavailable, degrading
    gracefully to a process-group-only kill.
    """
    result: list[int] = []
    stack = [pid]
    seen = {pid}
    while stack:
        cur = stack.pop()
        try:
            with open(f"/proc/{cur}/task/{cur}/children") as fh:
                kids = [int(x) for x in fh.read().split()]
        except OSError:
            continue
        for kid in kids:
            if kid not in seen:
                seen.add(kid)
                result.append(kid)
                stack.append(kid)
    return result


def _sigkill(pid: int) -> None:
    """SIGKILL one pid and its process group; swallow an already-gone process."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """SIGKILL *process* and every descendant.

    Claude Code runs Bash tools and sub-agents in their own sessions/process
    groups, so killing only the main group leaves them running. So:
      1. snapshot the whole descendant tree (while it is still intact);
      2. kill the PARENT first — a SIGKILL'd parent can't spawn anything new;
      3. kill every snapshotted descendant by pid.
    The window between snapshot and parent-kill is a few function calls (no
    awaits), so a child spawned in that gap is vanishingly unlikely.
    """
    pid = process.pid
    descendants = _descendant_pids(pid)   # 1. snapshot first
    _sigkill(pid)                         # 2. parent first (no respawns)
    try:
        process.kill()                   # let asyncio see the process is gone
    except (ProcessLookupError, OSError):
        pass
    for child in descendants:            # 3. the snapshotted children
        _sigkill(child)


# ----------------------------------------------------------------------
# Live progress (derived purely from the stream-json the CLI already emits —
# no extra LLM calls). The tracker accumulates events; the handler renders a
# snapshot at most once per PROGRESS_INTERVAL and a final summary at the end.
# ----------------------------------------------------------------------

# tool name -> (emoji, present-tense verb) for the live status line.
_TOOL_VERB: dict[str, tuple[str, str]] = {
    "Read": ("📖", "Reading"),
    "Edit": ("✏️", "Editing"),
    "Write": ("✏️", "Writing"),
    "MultiEdit": ("✏️", "Editing"),
    "NotebookEdit": ("✏️", "Editing"),
    "Bash": ("⚙️", "Running"),
    "Grep": ("🔍", "Searching"),
    "Glob": ("🔍", "Searching"),
    "WebFetch": ("🌐", "Fetching"),
    "WebSearch": ("🌐", "Searching"),
    "Task": ("🤖", "Subagent"),
    "Agent": ("🤖", "Subagent"),
    "TodoWrite": ("📝", "Planning"),
}
# Tools that mutate files — counted toward the "N files changed" tally.
_EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
# Tools that launch a subagent — the CLI uses "Task", this environment uses
# "Agent"; both carry subagent_type/description/prompt.
_SUBAGENT_TOOLS = frozenset({"Task", "Agent"})


def _fmt_duration(seconds: float) -> str:
    """Compact human duration: ``45s``, ``2m14s``, ``1h03m``."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def _count_lines(text: str) -> int:
    """Line count of *text* (0 for empty). Used for edit churn (+/−)."""
    return text.count("\n") + 1 if text else 0


def _edit_churn(name: str, inp: dict) -> tuple[int, int]:
    """Approximate (added, removed) line churn from an edit tool's input (no real diff)."""
    if name == "MultiEdit":
        added = removed = 0
        for e in inp.get("edits") or []:
            if isinstance(e, dict):
                removed += _count_lines(e.get("old_string") or "")
                added += _count_lines(e.get("new_string") or "")
        return added, removed
    if name == "Write":
        return _count_lines(inp.get("content") or ""), 0
    if name == "NotebookEdit":
        return _count_lines(inp.get("new_source") or ""), 0
    return _count_lines(inp.get("new_string") or ""), _count_lines(inp.get("old_string") or "")  # Edit


def _tool_result_text(block: dict) -> str:
    """Flatten a tool_result block's content (string or list of parts) to text."""
    content = block.get("content", "")
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


def _clip(text: str, limit: int) -> str:
    """Trim *text* to ~*limit* chars on a word boundary, adding '…' when cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return (text[:limit].rsplit(" ", 1)[0] or text[:limit]) + "…"


# The live status ticker posts as mrkdwn but shows Claude's own prose, which is
# standard Markdown. mrkdwn renders **bold** / [text](url) literally, so reduce
# inline Markdown to plain text — the ticker is an ephemeral status line, not
# formatted output. Only *paired* markers are stripped, so a lone underscore in
# snake_case or a spaced "*" survives. Order matters: links and code spans (which
# may contain "*") first, then bold before italic so "**" isn't read as two
# italic markers.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_MD_STRIKE_RE = re.compile(r"~~(.+?)~~")
_MD_ITALIC_RE = re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)")
# Tool errors arrive wrapped like ``<tool_use_error>…</tool_use_error>``; the angle
# brackets render as broken mrkdwn in Slack (they read as link syntax), so strip
# XML-ish tags before display. Only tag-shaped ``<word>`` — a lone ``a < b`` survives.
_TAG_RE = re.compile(r"</?[A-Za-z][\w-]*>")


def _strip_inline_md(text: str) -> str:
    """Reduce inline Markdown in *text* to plain text for the live status ticker."""
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = _MD_BOLD_RE.sub(r"\2", text)
    text = _MD_STRIKE_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\2", text)
    return text


# Keep the action log under Slack's per-block size ceiling — it now renders as a
# rich_text block (never collapsed), so this char cap is what bounds its height/size
# (the PROGRESS_ACTIONS count cap and this char cap both apply, whichever hits first).
_ACTIONS_CHARS = 2500
# Max todo lines shown in the live checklist before collapsing the tail.
_TODOS_MAX = 15
# Max subagents listed in the live status before collapsing the tail.
_SUBAGENTS_MAX = 10
# Max error lines listed in the live status' error block before collapsing.
_ERRORS_MAX = 5

# A material shrink in the live prompt size between consecutive real calls that we
# did NOT see a compact_boundary for = microcompaction: the CLI silently evicts
# stale tool outputs (file reads, shell output) to disk and emits no event, so the
# only signal is the drop. Counted as a "trim", shown apart from true compaction.
# The prompt is otherwise monotonic within a session (history only grows), so a
# drop past both thresholds is unambiguous; the floor just ignores tiny jitter.
_CTX_TRIM_RATIO = 0.9      # new prompt < 90% of the previous one, AND
_CTX_TRIM_FLOOR = 5_000    # at least this many tokens smaller

# Human meanings for the exit codes that otherwise read as cryptic numbers. Shell
# convention: 128+N means "killed by signal N", so anything in 129..192 without a
# specific entry is rendered as that. Everything else keeps the bare number.
_EXIT_MEANINGS = {
    1: "general error", 2: "usage error", 126: "not executable",
    127: "command not found", 128: "bad exit code", 130: "interrupted (Ctrl-C)",
    134: "aborted", 137: "killed — out of memory or SIGKILL", 139: "segfault",
    143: "terminated (SIGTERM)",
}
# A result that's ONLY an exit/error code — no message to describe it.
_BARE_CODE_RE = re.compile(r"(?i)^(?:exit code|exit status|error|command failed)[\s:]*"
                           r"(\d+)?[.\s]*$")
_EXIT_CODE_RE = re.compile(r"(?i)\bexit (?:code|status)\s*(\d+)")


def _exit_code_note(code: int) -> str:
    """A short human gloss for an exit code (empty if the number is self-evident)."""
    if code in _EXIT_MEANINGS:
        return _EXIT_MEANINGS[code]
    if 128 < code < 193:
        return f"killed by signal {code - 128}"
    return ""


def _describe_error(text: str, cause: str) -> str:
    """Turn a raw tool_result error into an informative one-liner.

    Prefixes the failing tool's context (``cause`` — "what ran") when the message
    on its own is a bare code, and glosses cryptic exit codes with their meaning.
    Never drops information: an already-descriptive message is kept as-is.
    """
    first = _strip_inline_md((text or "").strip().splitlines()[0]) if text.strip() else ""
    # Gloss "exit code N" -> "exit code N (meaning)" wherever it appears.
    m = _EXIT_CODE_RE.search(first)
    if m:
        note = _exit_code_note(int(m.group(1)))
        if note and "(" not in first:
            first = f"{first[:m.end()]} ({note}){first[m.end():]}"
    # A code-only line says nothing about WHAT failed — lead with the cause instead.
    if cause and (not first or _BARE_CODE_RE.match((text or "").strip().splitlines()[0])):
        first = f"{cause} → {first}" if first else cause
    return redact_secrets(_clip(first, 160))
# Status icon per subagent state (running / finished / finished-with-error).
_SUBAGENT_ICON = {"running": "🔄", "done": "✅", "error": "⚠️"}
# Max per-file lines in the changed-files section (Slack "Show more" collapses it).
_FILES_MAX = 99
# Status icon per todo state (no interactive checkboxes in a posted message).
_TODO_ICON = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}
# Context-window sizes (max input tokens) for the "ctx used/window" readout,
# keyed by model substring. Loaded from model_context.json at the repo root so
# they can be updated without a code change; the built-in map below is the
# fallback when the file is missing or malformed.
_MODEL_CONTEXT_CONFIG = Path(__file__).parent.parent / "model_context.json"
# Renders a diff as colored PNGs — the "phone version" offered in the Slack prompt.
_DIFF_PNG_TOOL = Path(__file__).resolve().parent.parent / "tools" / "diff_png.py"
_BUILTIN_MODEL_CONTEXT = {
    "haiku": 200_000,
    "fable": 1_000_000,
    "mythos": 1_000_000,
    "opus": 1_000_000,
    "sonnet": 1_000_000,
}


# USD per 1M tokens (input, output) by model substring; fallback when the file
# has no prices. Cache read ≈ 0.1× input, cache write ≈ 1.25× input.
_BUILTIN_MODEL_PRICES = {
    "haiku": (1.0, 5.0),
    "fable": (10.0, 50.0),
    "mythos": (10.0, 50.0),
    "opus": (5.0, 25.0),
    "sonnet": (3.0, 15.0),
}


def _load_model_context() -> tuple[dict[str, int], int]:
    """Return (window-by-substring, default) from model_context.json, or built-ins."""
    try:
        with open(_MODEL_CONTEXT_CONFIG) as f:
            cfg = json.load(f)
        windows = {str(k).lower(): int(v) for k, v in (cfg.get("windows") or {}).items()}
        default = int(cfg.get("default", 200_000))
        if windows:
            return windows, default
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("model_context.json unusable (%s) — using built-in windows.", exc)
    return dict(_BUILTIN_MODEL_CONTEXT), 200_000


def _load_model_prices() -> dict[str, tuple[float, float]]:
    """Return {substring: (in_price, out_price)} from model_context.json, or built-ins."""
    try:
        with open(_MODEL_CONTEXT_CONFIG) as f:
            cfg = json.load(f)
        prices = {
            str(k).lower(): (float(v["in"]), float(v["out"]))
            for k, v in (cfg.get("prices") or {}).items()
            if isinstance(v, dict) and "in" in v and "out" in v
        }
        if prices:
            return prices
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("model_context.json prices unusable (%s) — using built-ins.", exc)
    return dict(_BUILTIN_MODEL_PRICES)


_MODEL_CONTEXT, _DEFAULT_CONTEXT = _load_model_context()
_MODEL_PRICES = _load_model_prices()


def _fmt_tokens(n: int) -> str:
    """Compact token count: ``950``, ``4.2k``, ``18k``, ``1.2M``."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}k".replace(".0k", "k")
    return str(n)


def _short_model(model: str) -> str:
    """``claude-opus-4-8`` -> ``opus-4-8`` for the status line."""
    return model[len("claude-"):] if model.startswith("claude-") else model


def _model_window(model: str) -> int | None:
    """Context window for *model* by family substring, or None if no key matches."""
    m = (model or "").lower()
    for key, size in _MODEL_CONTEXT.items():
        if key in m:
            return size
    return None


def _model_context(model: str) -> int:
    return _model_window(model) or _DEFAULT_CONTEXT


def _model_price(model: str) -> "tuple[float, float] | None":
    """(input, output) USD per 1M tokens for *model*, or None if unknown."""
    m = (model or "").lower()
    for key, price in _MODEL_PRICES.items():
        if key in m:
            return price
    return None


def _estimate_cost(model: str, tok_in: int, tok_out: int, cache_r: int, cache_c: int) -> "float | None":
    """Rough USD cost from token usage; None when the model price is unknown.

    Cache reads bill ~0.1× input, cache writes ~1.25× input.
    """
    price = _model_price(model)
    if price is None:
        return None
    pin, pout = price
    return (tok_in * pin + tok_out * pout + cache_r * pin * 0.1 + cache_c * pin * 1.25) / 1_000_000


def _fmt_cost(usd: float) -> str:
    if usd < 0.01:
        return "<$0.01"
    if usd < 1:
        return f"${usd:.2f}"        # $0.35
    return f"${usd:.1f}"            # $22.1


class _Progress(NamedTuple):
    """An immutable progress snapshot handed to the daemon's ProgressCb.

    While running the daemon shows ``live`` (the last N actions, newest first)
    with ``meta`` rendered as a muted context line below it. ``summary`` is the
    collapsed end-state (one headline + an auto-collapsed detail block); ``done``
    tells the callback to render ``summary`` instead.
    """

    live: str
    summary: str
    done: bool
    meta: str = ""
    todos: str = ""  # checklist block (statuses), shown as its own section
    subagents: str = ""  # subagent list (purpose + status), shown as its own section
    errors: str = ""     # error detail block — live status only, not the reply
    files: str = ""  # changed-files block, shown as its own (collapsible) section


class _ProgressTracker:
    """Folds a stream of claude events into a renderable progress snapshot.

    Deterministic and side-effect-free (no clock of its own, no I/O): callers
    pass ``now`` so elapsed time is testable. ``dirty`` flags whether anything
    changed since the last ``snapshot`` so the throttle can skip no-op edits.
    Everything is derived from the events the CLI already emits — no LLM calls.
    """

    def __init__(self, start: float, prior_cost: float = 0.0) -> None:
        self._start = start
        # Unrounded USD already spent on EARLIER messages in this thread, so the
        # meta can show both this message's cost and the running thread total. Kept
        # as a float and summed before rounding — never add already-rounded dollars.
        self._prior_cost = prior_cost
        self._cap = PROGRESS_ACTIONS if PROGRESS_ACTIONS > 0 else 99
        # Rolling action history (oldest→newest), retained a bit beyond the cap.
        # (seq, text) pairs: each line carries the sequence number it was born
        # with, so removing a merged repeat's stale line never renumbers others.
        self._actions: deque[tuple[int, str]] = deque(maxlen=max(200, self._cap + 1))
        self._total = 0                      # actions ever recorded
        self._tool_count = 0
        self._files: list[str] = []          # ordered, unique edited/written paths
        self._file_set: set[str] = set()
        self._reads = 0
        self._commands: list[str] = []       # first line of each Bash command
        self._added = 0                      # lines added across edits (churn)
        self._removed = 0                    # lines removed across edits (churn)
        self._file_added: dict[str, int] = {}    # per-file added lines
        self._file_removed: dict[str, int] = {}  # per-file removed lines
        # Repeat-edit merger: full path -> {"n": edit count, "line": the exact
        # (seq, text) tuple currently in the deque for this file, "name": display
        # name}. Keyed by FULL path so a/x.py and b/x.py never merge; removal is
        # by tuple, and seq is unique, so it can never hit the wrong file's line.
        self._edits: dict[str, dict] = {}
        self._skills: list[str] = []         # skill names invoked (ordered, unique)
        self._mcp: set[str] = set()          # MCP server names used (mcp__<server>__*)
        self._compactions = 0                # times context was auto-compacted
        self._trims = 0                      # times context was microcompacted (drop, no event)
        self._expect_ctx_drop = False        # a compaction fired; its drop isn't a trim
        self._errors = 0                     # tool_result blocks flagged is_error
        self._error_lines: list[str] = []   # enriched first line of each error result
        self._tool_labels: dict[str, str] = {}  # tool_use_id -> "what ran" (for error context)
        self._tool_counts: dict[str, int] = {}  # per-tool-name use counts
        self._subagents = 0                  # Task tool launches (count)
        # Per-subagent runs: {id, type, desc, status} in launch order. Status
        # flips running -> done/error when the matching tool_result arrives.
        self._subagent_runs: list[dict] = []
        self._todos: list[tuple[str, str]] = []  # (content, status) latest TodoWrite
        self._turns = 0                      # assistant turns (overridden by result)
        self._usage: "dict | None" = None    # latest plan-quota payload (quota.get)
        self._model = ""                     # from system:init
        self._tok_in = 0
        self._tok_out = 0
        self._tok_cache_r = 0                # cache READ tokens (reused context)
        self._tok_cache_c = 0                # cache CREATION tokens
        self._ctx_last = 0                   # prompt size of the most recent call
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def ctx_last(self) -> int:
        return self._ctx_last

    def has_content(self) -> bool:
        """True once any action has been recorded (something worth posting).

        Gates the first status post so an empty "🔄 Working… · 0 tools" panel —
        model known but nothing done yet — never gets sent.
        """
        return self._total > 0

    def _push(self, action: str) -> "tuple[int, str]":
        """Record the newest action, collapsing consecutive duplicates.

        Secrets in the action text (command lines, URLs, prose) are masked here
        so nothing credential-shaped reaches Slack — see security.redact_secrets.
        Each line is stored as (seq, text) with seq stamped NOW — position-derived
        numbering would silently shift every older line when the repeat-edit
        merger removes one mid-deque. Returns the stored tuple for that merger.
        """
        action = redact_secrets(action)
        if self._actions and self._actions[-1][1] == action:
            return self._actions[-1]             # consecutive duplicate collapsed
        self._total += 1
        entry = (self._total, action)
        self._actions.append(entry)
        self._dirty = True
        return entry

    def ingest(self, event: Any) -> None:
        """Update state from one stream-json event."""
        if not isinstance(event, dict):
            return
        etype = event.get("type")
        if etype == "assistant":
            self._turns += 1
            message = event.get("message") or {}
            # "<synthetic>" assistant messages (CLI-injected, e.g. resume/compact
            # notices) report a near-zero prompt size. Folding their usage in would
            # clobber ctx_last to ~0% and make the live % jump up and down — so
            # skip them for both ctx and token totals.
            model = message.get("model")
            synthetic = model == "<synthetic>"
            if model and not synthetic:  # real id, e.g. claude-opus-4-8
                self._model = model
            usage = message.get("usage")
            if isinstance(usage, dict) and not synthetic:
                self._add_usage(usage)
            for block in message.get("content", []) or []:
                if isinstance(block, dict):
                    self._ingest_assistant_block(block)
        elif etype == "user":
            # tool_result blocks ride on user events: flip subagent status by
            # tool_use_id and count the failed ones.
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                is_err = bool(block.get("is_error"))
                self._mark_subagent_done(block.get("tool_use_id") or "", error=is_err)
                if is_err:
                    self._errors += 1
                    cleaned = _TAG_RE.sub("", _tool_result_text(block)).strip()
                    cause = self._tool_labels.get(block.get("tool_use_id") or "", "")
                    line = _describe_error(cleaned, cause)
                    if line:
                        self._error_lines.append(line)
                    self._dirty = True
        elif etype == "system":
            if event.get("model"):
                self._model = event["model"]
            if event.get("subtype") == "compact_boundary":  # CLI auto-compaction marker
                self._compactions += 1
                self._expect_ctx_drop = True   # the ensuing prompt drop is this compaction, not a trim
                self._dirty = True
        elif etype == "result":
            nt = event.get("num_turns")
            if isinstance(nt, int) and nt > 0:
                self._turns = nt
            usage = event.get("usage")
            if isinstance(usage, dict) and self._tok_out == 0:
                self._add_usage(usage)  # fallback when per-turn usage was absent
            self._dirty = True

    def _add_usage(self, u: dict) -> None:
        self._tok_in += u.get("input_tokens") or 0
        self._tok_out += u.get("output_tokens") or 0
        self._tok_cache_r += u.get("cache_read_input_tokens") or 0
        self._tok_cache_c += u.get("cache_creation_input_tokens") or 0
        # Full prompt size of THIS call = fresh input + cached (read + created).
        new_ctx = (
            (u.get("input_tokens") or 0)
            + (u.get("cache_read_input_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0)
        )
        # A material drop = context was reclaimed. If a compaction fired we already
        # counted it (and armed _expect_ctx_drop); otherwise it's a silent microcompaction.
        if (self._ctx_last and new_ctx < self._ctx_last * _CTX_TRIM_RATIO
                and (self._ctx_last - new_ctx) >= _CTX_TRIM_FLOOR):
            if self._expect_ctx_drop:
                self._expect_ctx_drop = False   # this drop belongs to the counted compaction
            else:
                self._trims += 1
        self._ctx_last = new_ctx
        self._dirty = True

    def _ingest_assistant_block(self, block: dict) -> None:
        btype = block.get("type")
        if btype == "tool_use":
            self._ingest_tool(block)
        elif btype == "text":
            text = (block.get("text") or "").strip()
            if text:
                self._push("💬 " + _clip(_strip_inline_md(text.splitlines()[0]), 400))
        elif btype == "thinking":
            if (block.get("thinking") or "").strip():
                self._push("🤔 Thinking…")
        elif btype == "compaction":  # API compaction content block
            self._compactions += 1
            self._expect_ctx_drop = True   # its prompt drop is this compaction, not a trim
            self._dirty = True

    def _ingest_tool(self, block: dict) -> None:
        name = block.get("name", "") or "Tool"
        inp = block.get("input", {}) or {}
        self._tool_count += 1
        self._tool_counts[name] = self._tool_counts.get(name, 0) + 1
        if name in _SUBAGENT_TOOLS:
            self._subagents += 1
            # description is the short purpose; fall back to the first prompt line.
            desc = (inp.get("description") or "").strip()
            if not desc and inp.get("prompt"):
                desc = str(inp["prompt"]).strip().splitlines()[0]
            self._subagent_runs.append({
                "id": block.get("id") or "",
                "type": (inp.get("subagent_type") or "").strip(),
                "desc": desc,
                "status": "running",
            })
        elif name == "TodoWrite":
            self._ingest_todos(inp)
        elif name == "Skill":
            sk = inp.get("skill") or inp.get("command") or inp.get("name")
            if sk:
                short = str(sk).split(":")[-1]  # "superpowers:brainstorming" -> "brainstorming"
                if short not in self._skills:
                    self._skills.append(short)
        if name.startswith("mcp__"):
            parts = name.split("__")  # mcp__<server>__<tool>
            if len(parts) >= 3:
                self._mcp.add(parts[1])
        emoji, verb = _TOOL_VERB.get(name, ("🔧", name))
        arg = ""
        label_arg = ""  # error-cause label: what ran, without ticker ×N decoration
        if name == "Skill":
            emoji, verb = "🧩", "Skill"
            arg = f"`{self._skills[-1]}`" if self._skills else ""
        elif name.startswith("mcp__"):  # 🔌 <tool> — server is in the MCP line
            emoji, verb = "🔌", name.split("__")[-1]
        elif name in _EDIT_TOOLS:
            path = inp.get("file_path") or inp.get("notebook_path") or ""
            a, r = _edit_churn(name, inp)
            self._added += a
            self._removed += r
            if path:
                if path not in self._file_set:
                    self._file_set.add(path)
                    self._files.append(path)
                self._file_added[path] = self._file_added.get(path, 0) + a
                self._file_removed[path] = self._file_removed.get(path, 0) + r
                # Repeat edits of ONE file merge to one ticker line that rides
                # newest: `foo.py` ×3 (+40/−12). Display only — _total still
                # counts every edit (via _push below), so numbering stays
                # monotone and "… and N more" absorbs the collapsed repeats.
                if path not in self._edits:      # first edit: pick the name once
                    self._edits[path] = {"n": 0, "line": None,
                                         "name": self._edit_name(path)}
                e = self._edits[path]
                e["n"] += 1
                arg = label_arg = f"`{e['name']}`"   # label: what ran, no ×N deco
                if e["n"] > 1:
                    fa, fr = self._file_added[path], self._file_removed[path]
                    churn = f" (+{fa}/−{fr})" if (fa or fr) else ""
                    arg += f" ×{e['n']}{churn}"
                    if e["line"] in self._actions:   # drop the stale line; may
                        self._actions.remove(e["line"])  # already be evicted/trimmed
        elif name == "Read":
            self._reads += 1
            path = inp.get("file_path") or ""
            arg = f"`{os.path.basename(path)}`" if path else ""
        elif name in _SUBAGENT_TOOLS:
            run = self._subagent_runs[-1] if self._subagent_runs else {}
            label = run.get("type") or "subagent"
            if run.get("desc"):
                label += f": {run['desc']}"
            arg = f"`{_clip(label, 120)}`"
        elif name == "Bash":
            cmd = (inp.get("command") or "").strip()
            if cmd:
                first = redact_secrets(cmd.splitlines()[0])
                self._commands.append(first)
                arg = f"`{_clip(first, 160)}`"
        elif name in ("Grep", "Glob"):
            pat = inp.get("pattern") or inp.get("query") or ""
            arg = f"`{_clip(pat, 120)}`" if pat else ""
        elif name in ("WebFetch", "WebSearch"):
            ref = inp.get("url") or inp.get("query") or ""
            arg = f"`{_clip(ref, 120)}`" if ref else ""
        # Remember what this tool ran, keyed by id, so a later error result can
        # name its cause ("`npm build` → exit 2") instead of a bare code.
        tool_id = block.get("id") or ""
        if tool_id:  # label = what the tool RAN — not the ticker's ×N display state
            self._tool_labels[tool_id] = _clip((f"{verb} {label_arg or arg}".strip() or name), 120)
        pushed = self._push(f"{emoji} {verb} {arg}".strip())
        if name in _EDIT_TOOLS:
            p = inp.get("file_path") or inp.get("notebook_path") or ""
            if p:  # remember the tuple as stored (post-redaction) for later removal
                self._edits[p]["line"] = pushed

    def _ingest_todos(self, inp: dict) -> None:
        """Capture the latest TodoWrite list (it replaces the whole list each call)."""
        parsed: list[tuple[str, str]] = []
        for t in inp.get("todos") or []:
            if isinstance(t, dict):
                content = t.get("content") or t.get("activeForm") or ""
                if content:
                    parsed.append((content, t.get("status") or "pending"))
        if parsed:
            self._todos = parsed
            self._dirty = True

    def _mark_subagent_done(self, tool_use_id: str, *, error: bool) -> None:
        """Flip the matching subagent run to done/error when its result arrives.

        Matched by tool_use_id, so non-Task tool_results never touch the list
        (only Task launches are recorded). A blank id matches nothing.
        """
        if not tool_use_id:
            return
        for run in self._subagent_runs:
            if run["id"] == tool_use_id and run["status"] == "running":
                run["status"] = "error" if error else "done"
                self._dirty = True
                return

    def _edit_name(self, path: str) -> str:
        """Display name for an edited file's ticker line.

        Plain basename normally; on a clash with a DIFFERENT tracked path,
        parent/basename; still clashing, the full path. Pure readability —
        two `x.py ×N` lines would be ambiguous to the reader. (Removal safety
        does NOT depend on this: the merger removes by (seq, text) tuple and
        seq is unique.) Chosen once (at the file's first edit) and kept stable.
        """
        taken = {e["name"] for p, e in self._edits.items() if p != path}
        name = os.path.basename(path)
        if name in taken:
            name = os.path.join(os.path.basename(os.path.dirname(path)), name)
        if name in taken:
            name = path
        return name

    # -- rendering helpers --------------------------------------------------

    def _tool_breakdown(self) -> str:
        """Per-tool counts, busiest first: ``8 read · 5 edit · 3 bash``.

        MCP tools (``mcp__server__tool``) collapse to a single ``mcp`` bucket so
        their long names don't blow up the line — the servers are listed separately.
        """
        if not self._tool_counts:
            return "0 tools"
        grouped: dict[str, int] = {}
        for name, n in self._tool_counts.items():
            key = "mcp" if name.startswith("mcp__") else name.lower()
            grouped[key] = grouped.get(key, 0) + n
        ordered = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))
        return " · ".join(f"{n} {key}" for key, n in ordered[:6])

    def _files_block(self) -> str:
        """Header + a bulleted list of changed files with per-file churn.

        A plain mrkdwn section (keeps `code`/bold) — a long one gets Slack's own
        "Show more" collapse. Per-file +/− is churn, not a real diff (Slack has no
        nested collapsible blocks).
        """
        if not self._files:
            return ""
        n = len(self._files)
        churn = f" (+{self._added}/−{self._removed})" if (self._added or self._removed) else ""
        lines = [f"📝 *{n} file{'s' if n != 1 else ''} changed*{churn}"]
        for path in self._files[:_FILES_MAX]:
            a = self._file_added.get(path, 0)
            r = self._file_removed.get(path, 0)
            per = f"  +{a}/−{r}" if (a or r) else ""
            lines.append(f"• `{os.path.basename(path)}`{per}")
        if n > _FILES_MAX:
            lines.append(f"…and {n - _FILES_MAX} more")
        return "\n".join(lines)

    def _todos_block(self) -> str:
        """Checklist of the current todos with status icons (empty when none).

        Rendered in the agent's own TodoWrite order (execution order) and numbered
        from the top, so the item worked first reads as ``1.`` — not reversed.
        """
        if not self._todos:
            return ""
        done = sum(1 for _, s in self._todos if s == "completed")
        lines = [f"📋 *Todos {done}/{len(self._todos)}*"]
        for i, (content, status) in enumerate(self._todos[:_TODOS_MAX], start=1):
            icon = _TODO_ICON.get(status, "⬜")
            lines.append(f"`{i}.` {icon} {_clip(_strip_inline_md(content), 80)}")
        if len(self._todos) > _TODOS_MAX:
            lines.append(f"…+{len(self._todos) - _TODOS_MAX} more")
        return "\n".join(lines)

    def _subagents_block(self) -> str:
        """List launched subagents with their purpose and status (empty when none).

        Status is launch/running/done only — the parent stream does not expose a
        subagent's internal tool activity, so the bridge cannot show it.
        """
        if not self._subagent_runs:
            return ""
        done = sum(1 for r in self._subagent_runs if r["status"] != "running")
        n = len(self._subagent_runs)
        lines = [f"🤖 *Subagents {done}/{n}*"]
        for run in self._subagent_runs[:_SUBAGENTS_MAX]:
            icon = _SUBAGENT_ICON.get(run["status"], "🔄")
            label = run["type"] or "subagent"
            if run["desc"]:
                label += f" — {run['desc']}"
            lines.append(f"{icon} {_clip(_strip_inline_md(label), 80)}")
        if n > _SUBAGENTS_MAX:
            lines.append(f"…+{n - _SUBAGENTS_MAX} more")
        return "\n".join(lines)

    def _errors_block(self) -> str:
        """Non-blocking tool errors with their text — live status only (empty when none).

        These are never terminal — they're friction the agent saw and worked
        around, worth surfacing as a hint that tooling or instructions could be
        better. Deliberately plain (no icon, no colour): a calm, informative log,
        not an alarm. The dimmed meta shows just the count; the detail lives here
        so it doesn't ride on the final reply.
        """
        if not self._errors:
            return ""
        lines = [f"*{self._errors} error{'s' if self._errors != 1 else ''}*"]
        for line in self._error_lines[-_ERRORS_MAX:]:
            lines.append(f"• {line}")
        return "\n".join(lines)

    def run_cost(self) -> "float | None":
        """Unrounded USD for THIS message's tokens (None when the price is unknown)."""
        return _estimate_cost(
            self._model, self._tok_in, self._tok_out, self._tok_cache_r, self._tok_cache_c
        )

    def _tokens_str(self) -> str:
        s = (
            f"🪙 ↑{_fmt_tokens(self._tok_in)} ↓{_fmt_tokens(self._tok_out)}"
            f" ⚡{_fmt_tokens(self._tok_cache_r)}"
        )
        cost = self.run_cost()
        if cost is not None:
            # Sum the unrounded floats, then round ONCE for each figure — adding
            # already-rounded per-message dollars would drift the thread total.
            if self._prior_cost > 0:
                total = self._prior_cost + cost
                s += f" ({_fmt_cost(cost)} msg · {_fmt_cost(total)} total)"
            else:                                # first message: msg == total
                s += f" ({_fmt_cost(cost)})"   # 🪙 ↑12k ↓4.2k ⚡368k ($22)
        return s

    def _ctx_pct(self) -> "int | None":
        """Latest prompt size as a % of the model's context window (None if unknown)."""
        if not self._ctx_last:
            return None
        return round(self._ctx_last / _model_context(self._model) * 100)

    def _meta(self, elapsed: str) -> str:
        """Muted context block: activity line + resource line."""
        # Lead with the total action count, then the per-tool breakdown.
        head = f"{self._tool_count} tool{'s' if self._tool_count != 1 else ''}"
        if self._tool_counts:
            head += f" ({self._tool_breakdown()})"
        line_a: list[str] = [head]
        if self._files:  # files changed + line churn, mirroring the files block header
            n = len(self._files)
            churn = f" (+{self._added}/−{self._removed})" if (self._added or self._removed) else ""
            line_a.append(f"{n} file{'s' if n != 1 else ''} changed{churn}")
        if self._subagents:
            line_a.append(f"🤖 {self._subagents} subagent{'s' if self._subagents != 1 else ''}")
        if self._compactions:
            line_a.append(f"🗜️ compacted {self._compactions}×")
        if self._trims:  # microcompaction (silent tool-output eviction), inferred from ctx drop
            line_a.append(f"♻️ trimmed {self._trims}×")
        if self._errors:  # count only here; the text lives in the live error block
            line_a.append(f"{self._errors} error{'s' if self._errors != 1 else ''}")
        # elapsed lives on the resource line (with turns/model) so the activity line
        # doesn't outgrow it once files/compacted/trimmed/errors pile on.

        line_b: list[str] = []
        if self._tok_out or self._tok_in:
            line_b.append(self._tokens_str())
        pct = self._ctx_pct()
        if pct is not None:  # icon + used/window + % (window is per-model, so % stays sane)
            window = _model_context(self._model)
            ctx_str = f"{_ctx_icon(pct)} ctx {_fmt_tokens(self._ctx_last)}/{_fmt_tokens(window)} ({pct}%)"
            if CTX_WARN_PCT > 0 and pct >= CTX_WARN_PCT:  # in the meta so it rides on the reply
                ctx_str += " — consider a new session"
            line_b.append(ctx_str)
        if self._turns:
            line_b.append(f"{self._turns} turn{'s' if self._turns != 1 else ''}")
        line_b.append(elapsed)
        if self._model:
            line_b.append(_short_model(self._model))
        q = quota.format_status(self._usage, _ctx_icon)  # plan quota: 🟠 wk 52% · Fable 70%
        if q:
            line_b.append(q)

        line_c: list[str] = []
        if self._skills:
            line_c.append("🧩 " + ", ".join(self._skills[:6]))
        if self._mcp:
            line_c.append("🔌 " + ", ".join(sorted(self._mcp)[:6]))

        lines = [" · ".join(line_a)]
        if line_b:
            lines.append(" · ".join(line_b))
        if line_c:
            lines.append(" · ".join(line_c))
        return "\n".join(lines)

    def snapshot(self, now: float, *, done: bool = False, idle: float = 0.0) -> _Progress:
        """Render the current state; clears ``dirty``.

        Shows the most recent actions (up to PROGRESS_ACTIONS), newest first;
        anything older collapses into "… and N more". *idle* (seconds since the
        last stream event) surfaces an "⏳ idle" hint at the top when quiet.
        """
        self._dirty = False
        elapsed = _fmt_duration(now - self._start)
        candidates = list(reversed(self._actions))  # newest first
        shown: list[str] = []
        used = 0
        # Numbers are stamped at push time (newest = highest), so the count is
        # visible, new actions clearly append at the top, and a merged repeat's
        # removal never renumbers the lines around it.
        for seq, line in candidates[:self._cap]:
            numbered = f"`{seq}.` {line}"
            if shown and used + len(numbered) + 1 > _ACTIONS_CHARS:
                break                            # keep the block under Slack's limit
            shown.append(numbered)
            used += len(numbered) + 1
        hidden = max(0, self._total - len(shown))   # everything not shown
        lines = list(shown)
        if hidden:
            lines.append(f"… and {hidden} more")
        if not done and idle >= IDLE_HINT:       # surface a stuck/quiet run
            lines.insert(0, f"⏳ idle {_fmt_duration(idle)} — still on the current step")
        # The context nudge lives in the meta line (so it rides on the final reply,
        # not only the live status that's removed on completion).
        live = "\n".join(lines) if lines else "🔄 Working…"
        return _Progress(
            live=live, meta=self._meta(elapsed), todos=self._todos_block(),
            subagents=self._subagents_block(), errors=self._errors_block(),
            files=self._files_block(), summary=self._render_summary(elapsed), done=done,
        )

    def _render_summary(self, elapsed: str) -> str:
        """One-line "✅ Done" headline. The full detail stays in the live sections,
        which the daemon keeps below this headline on completion (not collapsed)."""
        head_parts: list[str] = []
        if self._files:
            head_parts.append(
                f"{len(self._files)} file{'s' if len(self._files) != 1 else ''} changed"
            )
        if self._added or self._removed:
            head_parts.append(f"+{self._added}/−{self._removed}")
        head_parts.append(f"{self._tool_count} tool{'s' if self._tool_count != 1 else ''}")
        if self._errors:
            head_parts.append(f"{self._errors} error{'s' if self._errors != 1 else ''}")
        head_parts.append(elapsed)
        return "✅ Done · " + " · ".join(head_parts)


class ClaudeHandler:
    """
    Manages Claude Code CLI invocations for Slack messages.

    Args:
        slack_client: An async Slack WebClient (``self._app.client``).
    """

    def __init__(self, slack_client: Any, *, store_path: "Path | None" = None) -> None:
        self._slack_client = slack_client
        self._store_path = store_path or session_store.SESSIONS_PATH
        self._bot_user_id: str = ""
        self._sessions: dict[str, str] = {}  # thread_ts → session UUID
        self._project_map: dict[str, Any] = _load_project_map()
        # Resolved at startup: channel ID → {"path": str|None, "plugin_dir": str|None,
        #                                    "worktrees": dict[str, str]}
        self._channel_id_to_project: dict[str, dict] = {}
        # thread_ts → (cwd, plugin_dir) chosen when the thread started, so
        # replies stay in the same worktree without re-tagging.
        self._thread_config: dict[str, tuple[str | None, str | None]] = {}
        # Feature C: in-memory tracking of in-flight Flow-B subprocesses so a
        # 🛑 reaction can kill the right run. thread_ts → live subprocess.
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        # thread_ts values the user stopped, so the daemon can suppress the
        # (partial) normal reply for that run.
        self._stopped: set[str] = set()
        # thread_ts → the thread's model pick, and → prompt size of its last
        # finished run (the model-switch context guard). Both persisted.
        self._thread_model: dict[str, str] = {}
        self._thread_ctx: dict[str, int] = {}

    async def initialize(self) -> None:
        """Cache the bot's own user ID and resolve channel names to IDs."""
        resp = await self._slack_client.auth_test()
        self._bot_user_id = resp["user_id"]
        logger.info("ClaudeHandler initialized, bot_user_id=%s", self._bot_user_id)

        for thread_ts, rec in session_store.load(self._store_path).items():
            sid = rec.get("session_id")
            if sid:
                self._sessions[thread_ts] = sid
            self._thread_config[thread_ts] = (rec.get("cwd"), rec.get("plugin_dir"))
            if rec.get("model"):
                self._thread_model[thread_ts] = rec["model"]
            if rec.get("ctx_tokens"):
                self._thread_ctx[thread_ts] = rec["ctx_tokens"]

        if self._project_map:
            await self._resolve_channel_ids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _maybe_ctx_nudge(self, thread_ts: str | None, prompt: str) -> str:
        """Append the context note so the agent raises a fresh-session recommendation
        in its own reply, gated on the thread's last-known ``ctx_pct``.

        While ctx >= CTX_WARN_PCT the note always carries the CHECKPOINT clause
        (the agent decides whether it's a good moment). The URGENCY clause is added
        only when context has crossed into a worse severity band than last time
        (🟠→🔴→🚨), i.e. once per escalation; ``ctx_nudged`` stores that band.
        """
        if not (CTX_NUDGE_AGENT and CTX_WARN_PCT > 0 and thread_ts):
            return prompt
        rec = session_store.load(self._store_path).get(thread_ts) or {}
        pct = rec.get("ctx_pct")
        if not isinstance(pct, int) or pct < CTX_WARN_PCT:  # not in a recommended zone
            return prompt
        band = _ctx_band(pct)
        escalated = band > int(rec.get("ctx_nudged") or 0)
        if escalated:
            session_store.upsert(thread_ts, ctx_nudged=band, path=self._store_path)
        note = "\n\n[context: " + _CTX_NOTE_HEAD.format(pct=pct)
        if escalated:
            note += _CTX_NOTE_URGENCY
        note += _CTX_NOTE_CHECKPOINT + "]"
        logger.info(
            "ctx nudge appended for thread %s at %d%% (band %d, escalated=%s)",
            thread_ts, pct, band, escalated,
        )
        return prompt + note

    async def handle_message(
        self, channel: str, message_ts: str, text: str,
        files: list[tuple[str, str]] | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> str:
        """Handle a new top-level Slack message (start a new Claude session)."""
        label, text = _parse_worktree_tag(text)
        project_dir, plugin_dir = self._get_project_config(channel, label)

        session_id = str(uuid.uuid4())
        self._sessions[message_ts] = session_id
        self._thread_config[message_ts] = (project_dir, plugin_dir)
        session_store.upsert(
            message_ts, session_id=session_id, cwd=project_dir,
            plugin_dir=plugin_dir, in_flight=False, pid=None,
            path=self._store_path,
        )
        logger.info("New Claude session %s for thread %s", session_id, message_ts)

        prompt = _append_attachment_note(text, files or [])
        return await self._run_session(
            prompt, session_id=session_id, resume=False, plugin_dir=plugin_dir,
            cwd=project_dir, thread_ts=message_ts, channel=channel, progress_cb=progress_cb,
        )

    async def handle_thread_reply(
        self, channel: str, thread_ts: str, text: str,
        files: list[tuple[str, str]] | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> str:
        """Handle a threaded reply: resume the real session, else scrape once.

        Policy ("scrape once, then resume forever"):
          1. session_id present AND its jsonl exists  -> --resume; on failure
             retry once; if it still fails -> return the error and keep the
             session (login/overload/CLI errors would sink a new session too).
          2. session_id present but jsonl missing      -> scrape fallback.
          3. no session_id                             -> scrape fallback.
        """
        session_id = self._sessions.get(thread_ts)
        # Thread inherits the worktree chosen at start; re-tagging mid-thread
        # would be confusing, so we don't re-parse here. Falls back to default
        # config only if the thread state was lost (container restart).
        project_dir, plugin_dir = self._thread_config.get(thread_ts) or self._get_project_config(channel)

        # Self-heal a stale cache: if the thread's dir has gone missing (e.g. a
        # projects.json path was wrong, then fixed + reloaded), re-resolve from
        # the current mapping instead of spawning into a dir that can't run.
        if project_dir is not None and not os.path.isdir(project_dir):
            fresh_dir, fresh_plugin = self._get_project_config(channel)
            if fresh_dir is not None and os.path.isdir(fresh_dir):
                logger.warning(
                    "Thread %s cwd %s is gone — re-resolved to %s after reload.",
                    thread_ts, project_dir, fresh_dir,
                )
                project_dir, plugin_dir = fresh_dir, fresh_plugin
                self._thread_config[thread_ts] = (project_dir, plugin_dir)
                session_store.upsert(
                    thread_ts, cwd=project_dir, plugin_dir=plugin_dir,
                    path=self._store_path,
                )

        if session_id and _jsonl_path(project_dir, session_id).exists():
            logger.info("Resuming session %s for thread %s", session_id, thread_ts)
            prompt = self._maybe_ctx_nudge(thread_ts, _append_attachment_note(text, files or []))
            reply = await self._run_session(
                prompt, session_id=session_id, resume=True, plugin_dir=plugin_dir,
                cwd=project_dir, thread_ts=thread_ts, channel=channel, progress_cb=progress_cb,
            )
            if not _is_run_failure(reply):
                return reply
            logger.warning("Resume of %s failed; retrying once.", session_id)
            reply = await self._run_session(
                prompt, session_id=session_id, resume=True, plugin_dir=plugin_dir,
                cwd=project_dir, thread_ts=thread_ts, channel=channel, progress_cb=progress_cb,
            )
            if _is_run_failure(reply):
                logger.warning("Resume of %s failed twice; keeping the session.", session_id)
            return reply
        else:
            logger.info(
                "No resumable session for thread %s (session_id=%s) — scraping.",
                thread_ts, session_id,
            )

        return await self._scrape_and_run(
            channel, thread_ts, project_dir, plugin_dir, files, progress_cb=progress_cb,
        )

    async def _scrape_and_run(
        self, channel: str, thread_ts: str, project_dir: str | None, plugin_dir: str | None,
        files: list[tuple[str, str]] | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> str:
        """Last-resort fallback: replay scraped thread history under a NEW session.

        Mints a fresh ``--session-id``, runs the scraped prompt, and persists the
        new id so subsequent replies take the hot --resume path (step 1).
        """
        prompt = await self._build_thread_prompt(channel, thread_ts)
        prompt = _append_attachment_note(prompt, files or [])
        new_id = str(uuid.uuid4())
        reply = await self._run_session(
            prompt, session_id=new_id, resume=False, plugin_dir=plugin_dir,
            cwd=project_dir, thread_ts=thread_ts, channel=channel, progress_cb=progress_cb,
        )
        if _is_run_failure(reply):
            return reply  # the new session never ran; don't swap it in
        self._sessions[thread_ts] = new_id
        self._thread_config[thread_ts] = (project_dir, plugin_dir)
        session_store.upsert(
            thread_ts, session_id=new_id, cwd=project_dir, plugin_dir=plugin_dir,
            path=self._store_path,
        )
        if reply not in (_TIMEOUT_REPLY, _INTERRUPTED_REPLY):
            reply = _NEW_SESSION_NOTE + reply
        return reply

    def _effective_model(self, thread_ts: str, channel: str) -> str | None:
        """The ``--model`` for this thread: thread pick → channel pick → CLAUDE_MODEL → none.

        A channel pick is skipped for a thread whose context won't fit its window.
        """
        channel_model = self._channel_model(channel)
        if channel_model and not self._ctx_fits(thread_ts, channel_model):
            channel_model = None
        return (
            self._thread_model.get(thread_ts)
            or channel_model
            or os.environ.get("CLAUDE_MODEL")
            or None
        )

    def _channel_model(self, channel: str) -> str | None:
        return (self._channel_id_to_project.get(channel) or {}).get("model")

    def _ctx_fits(self, thread_ts: str, model: str) -> bool:
        """False when the thread's last context is bigger than *model*'s known window."""
        window = _model_window(model)
        return not window or self._thread_ctx.get(thread_ts, 0) <= window

    def _set_thread_model(self, thread_ts: str, model: str | None) -> None:
        if model:
            self._thread_model[thread_ts] = model
        else:
            self._thread_model.pop(thread_ts, None)
        session_store.upsert(thread_ts, model=model, path=self._store_path)

    async def _run_session(
        self, prompt: str, *, session_id: str, resume: bool, plugin_dir: str | None,
        cwd: str | None, thread_ts: str, channel: str, progress_cb: ProgressCb | None,
    ) -> str:
        """Run *prompt* on the thread's effective model, applying ``@@model`` picks.

        A reply that records a pick only acknowledged it (the system prompt says
        so), so it is dropped and the same message is re-run once on the same
        session with the — possibly unchanged — effective ``--model``; markers on
        the re-run are applied but never re-run again. Notes on refused picks
        are kept either way. A re-run that fails twice undoes a pick that changed
        the model, so a model the CLI can't run (e.g. a made-up ``claude-…`` id)
        can't lock the thread.
        """
        model = self._effective_model(thread_ts, channel)
        if resume:
            cmd = self._build_cmd(resume=session_id, plugin_dir=plugin_dir, model=model)
        else:
            cmd = self._build_cmd(session_id=session_id, plugin_dir=plugin_dir, model=model)
        reply = await self._run_claude(
            cmd, prompt, cwd=cwd, thread_ts=thread_ts, channel=channel, progress_cb=progress_cb,
        )
        prev_thread, prev_channel = self._thread_model.get(thread_ts), self._channel_model(channel)
        text, notes, picked = await self._apply_model_markers(reply, thread_ts, channel)
        if not picked:
            return _join_notes(text, notes)
        # The old run only acknowledged the pick; answer the same message after it.
        new_model = self._effective_model(thread_ts, channel)
        logger.info("Thread %s picked a model (%s → %s); re-running.", thread_ts, model, new_model)
        cmd = self._build_cmd(resume=session_id, plugin_dir=plugin_dir, model=new_model)
        prompt += _SWITCH_NOTE.format(model=new_model or "the default model")
        for _ in range(2):  # retry once: a transient failure shouldn't undo a good pick
            reply = await self._run_claude(
                cmd, prompt, cwd=cwd, thread_ts=thread_ts, channel=channel,
                progress_cb=progress_cb,
            )
            if not _is_run_failure(reply):
                text, more, _ = await self._apply_model_markers(reply, thread_ts, channel)
                return _join_notes(text, notes + more)
        # The bridge note goes first, so the resume policy doesn't retry this reply.
        if new_model == model:
            return _join_notes(_RERUN_FAILED + "\n" + reply, notes)
        logger.warning("Thread %s can't run on %s; undoing the switch.", thread_ts, new_model)
        if self._thread_model.get(thread_ts) != prev_thread:
            self._set_thread_model(thread_ts, prev_thread)
        if self._channel_model(channel) != prev_channel:
            await self._set_channel_model(channel, prev_channel)
        failed = _SWITCH_FAILED.format(
            model=new_model or "the default model", old=model or "the default model",
        )
        return _join_notes(failed + "\n" + reply, notes)

    async def _apply_model_markers(
        self, reply: str, thread_ts: str, channel: str,
    ) -> tuple[str, list[str], bool]:
        """Strip ``@@model`` lines from *reply* and record each pick.

        Returns ``(text, notes, picked)``: the reply without marker lines, one
        plain sentence per refused pick (unknown name, thread context bigger
        than the target's window, a channel pick that can't be stored — none of
        which changes anything), and whether any pick was recorded. A channel
        pick too small for this thread's context is still recorded for the
        channel, with a note that this thread stays put.
        """
        text, picks = _parse_model_markers(reply)
        if not picks:
            return reply, [], False
        notes: list[str] = []
        picked = False
        for scope, name in picks:
            model = None if name == "default" else name
            if model and not _MODEL_NAME_RE.match(model):
                notes.append(_NOT_A_MODEL.format(name=name))
                continue
            fits = model is None or self._ctx_fits(thread_ts, model)
            sizes = {} if fits else {
                "model": model, "ctx": _fmt_tokens(self._thread_ctx[thread_ts]),
                "window": _fmt_tokens(_model_window(model)),
            }
            if scope == "thread":
                if not fits:
                    notes.append(_CTX_TOO_BIG.format(**sizes))
                    continue
                self._set_thread_model(thread_ts, model)
            else:
                refused = await self._set_channel_model(channel, model)
                if refused:
                    notes.append(refused)
                    continue
                if not fits and thread_ts not in self._thread_model:
                    notes.append(_CTX_TOO_BIG_HERE.format(**sizes))
            picked = True
        return text, notes, picked

    async def _set_channel_model(self, channel: str, model: str | None) -> str | None:
        """Write (or, for None, remove) ``"model"`` on the channel's projects.json entry.

        Applied live via ``reload_projects``. Returns None once written, or a
        note for the user when nothing changed: the channel has no projects.json
        entry, or the file can't be read (e.g. malformed) or written.
        """
        key = (self._channel_id_to_project.get(channel) or {}).get("key")
        if not key:
            return _NO_CHANNEL_ENTRY
        try:
            mapping = _load_project_map()
            if key not in mapping:
                return _NO_CHANNEL_ENTRY
            entry = mapping[key]
            if isinstance(entry, str):
                entry = {"path": entry}
            if model:
                entry["model"] = model
            else:
                entry.pop("model", None)
            # A plain-string entry collapses back once only its path is left.
            mapping[key] = entry["path"] if set(entry) == {"path"} else entry
            # Atomic temp + fsync + replace, keeping the file's mode and non-ASCII keys.
            session_store.save(
                mapping, PROJECTS_CONFIG, ensure_ascii=False,
                mode=stat.S_IMODE(PROJECTS_CONFIG.stat().st_mode),
            )
        except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError
            logger.warning("Couldn't set channel %s model in %s: %s", channel, PROJECTS_CONFIG, exc)
            return _PROJECTS_UNWRITABLE
        await self.reload_projects()
        return None

    async def stop(self, thread_ts: str) -> bool:
        """Kill the in-flight Flow-B subprocess for *thread_ts*, if any.

        Marks the thread stopped so the daemon suppresses the partial reply.
        Returns True when a tracked process was found and killed, else False.
        """
        process = self._processes.get(thread_ts)
        if process is None:
            return False
        self._stopped.add(thread_ts)
        logger.info("Stopping Claude subprocess for thread %s", thread_ts)
        _kill_process_tree(process)
        await process.wait()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_project_config(
        self, channel_id: str, label: str | None = None
    ) -> tuple[str | None, str | None]:
        """Return (project_dir, plugin_dir) for a Slack channel ID.

        When *label* is provided and matches a registered worktree for the
        channel, the worktree path is returned instead of the default. An
        unknown label falls back to the default with a warning so messages
        aren't silently dropped.

        Both values are ``None`` when no mapping exists for the channel.
        """
        config = self._channel_id_to_project.get(channel_id)
        if not config:
            logger.info("No project mapping for channel %s — using default cwd.", channel_id)
            return None, None

        plugin_dir = config["plugin_dir"]
        worktrees: dict[str, str] = config.get("worktrees", {})
        default_path = config["path"]

        if label and label in worktrees:
            return worktrees[label], plugin_dir

        if label and default_path:
            dynamic = _resolve_dynamic_worktree(default_path, label)
            if dynamic:
                return dynamic, plugin_dir

        path = default_path
        logger.info(
            "Channel %s → project %s%s",
            channel_id, path,
            f" (plugin_dir={plugin_dir})" if plugin_dir else "",
        )
        return path, plugin_dir

    async def reload_projects(self) -> int:
        """Re-read projects.json and re-resolve channel IDs, live.

        Lets a ``projects.json`` edit take effect without restarting the daemon,
        so the Slack connection and any in-flight runs are left untouched. It is
        triggered two ways (see docs/reloading-projects.md):

          * the ``RELOAD`` socket verb, sent by ``reloadctl.py`` (which replies
            with the channel count so the operator gets confirmation), and
          * a ``SIGHUP`` to the daemon (the ops idiom, e.g. ``pm2 sendSignal``).

        A live thread keeps the ``(cwd, plugin_dir)`` cached in
        ``_thread_config`` when it started, so a reload never relocates a
        running conversation — with one safe exception: if that cached dir has
        since gone missing, the next reply re-resolves it from the current
        mapping (see ``handle_thread_reply``), since a missing dir can't run
        anyway. If the re-resolve fails (e.g. a Slack API error), the previous
        mapping is preserved.

        Returns the number of channels mapped after the reload. Mapped paths
        that don't exist on disk are logged as warnings so an operator sees the
        mistake at reload time, before any message hits the bad path.
        """
        self._project_map = _load_project_map()
        if self._project_map:
            await self._resolve_channel_ids()
        else:
            self._channel_id_to_project = {}
        count = len(self._channel_id_to_project)
        missing = self.missing_project_dirs()
        if missing:
            logger.warning(
                "Reload: %d channel(s) map to a missing directory: %s",
                len(missing), ", ".join(p for _, p in missing),
            )
        logger.info("Reloaded project map: %d channel(s) mapped.", count)
        return count

    def missing_project_dirs(self) -> list[tuple[str, str]]:
        """``(channel_id, path)`` for mapped channels whose path is not a dir.

        Read-only preflight over the resolved mapping: a channel resolves fine
        (name→ID) yet still point at a path that doesn't exist on disk, which
        would fail only later when a message tries to spawn there. Surfacing it
        at reload time turns a confusing mid-run error into an obvious one.
        """
        return [
            (cid, cfg["path"])
            for cid, cfg in self._channel_id_to_project.items()
            if cfg.get("path") and not os.path.isdir(cfg["path"])
        ]

    async def _resolve_channel_ids(self) -> None:
        """Resolve channel names from project_map to Slack channel IDs.

        Builds the mapping in a local dict and swaps it in atomically at the
        end, so a message arriving mid-reload never observes a partial (or
        empty) mapping, and a mid-way failure leaves the previous one intact.
        """
        try:
            result = await self._slack_client.conversations_list(
                types="public_channel,private_channel", limit=1000,
            )
            channels = result.get("channels", [])

            name_to_id: dict[str, str] = {}
            for ch in channels:
                name_to_id[f"#{ch['name']}"] = ch["id"]
                name_to_id[ch["name"]] = ch["id"]
                name_to_id[ch["id"]] = ch["id"]  # allow raw IDs in config

            resolved: dict[str, dict] = {}
            for channel_key, value in self._project_map.items():
                # Normalise the legacy string format and the dict format.
                # "key" is the raw projects.json key, so a channel pick can be written back.
                if isinstance(value, str):
                    config = {
                        "path": value, "plugin_dir": None, "worktrees": {},
                        "model": None, "key": channel_key,
                    }
                else:
                    model = value.get("model")
                    if model is not None and not (isinstance(model, str) and _MODEL_NAME_RE.match(model)):
                        logger.warning(
                            "Channel %s: ignoring invalid model %r in projects.json.", channel_key, model,
                        )
                        model = None
                    config = {
                        "path": value.get("path"),
                        "plugin_dir": value.get("plugin_dir"),
                        "worktrees": value.get("worktrees") or {},
                        "model": model,
                        "key": channel_key,
                    }

                # DM channel IDs (D...) and raw channel IDs (C...) are not
                # returned by conversations_list — register them directly.
                if channel_key.startswith(("C", "D")) and channel_key not in name_to_id:
                    resolved[channel_key] = config
                    logger.info(
                        "Mapped %s (raw ID) → %s%s",
                        channel_key, config["path"],
                        f" plugin_dir={config['plugin_dir']}" if config["plugin_dir"] else "",
                    )
                    continue

                channel_id = name_to_id.get(channel_key)
                if channel_id:
                    resolved[channel_id] = config
                    logger.info(
                        "Mapped %s (ID: %s) → %s%s",
                        channel_key, channel_id, config["path"],
                        f" plugin_dir={config['plugin_dir']}" if config["plugin_dir"] else "",
                    )
                else:
                    logger.warning("Channel %s not found in workspace — skipping.", channel_key)

            self._channel_id_to_project = resolved

        except Exception as exc:
            logger.error("Failed to resolve channel IDs: %s", exc)

    # Flow-B Claude is spawned by the bridge itself (``claude -p``), so the
    # ``claude-slack-bridge`` MCP server that gives terminal sessions
    # ``ask_on_slack`` is not there. Skills and CLAUDE.md rules that name it
    # must fall back to a plain reply. Other MCP servers in .mcp.json load
    # normally.
    _FLOW_B_SYSTEM_PROMPT = (
        "You are replying in a Slack thread: your reply text is posted there, and "
        "the user's next reply resumes this session. Never call "
        "mcp__claude-slack-bridge__ask_on_slack; if a skill or command says to use "
        "it, end your turn with that message as your reply instead. Don't mention "
        "MCP, tool availability or the claude-slack-bridge server. "
        "To send a file, put '@@attach <absolute path>' on a line of its own (one "
        "line per file; the path must exist on disk). The bridge uploads the file "
        "and removes that line, so put any explanation on other lines. "
        "Show code or text changes in a fenced ```diff block, one per file with its "
        "path in bold above it: '-' old lines, '+' new lines, a leading space for "
        "1-3 unchanged context lines, changed parts only. Phones don't color it, so "
        "after a big diff you may offer a phone version in one short line. Only if "
        "the user asks: write the unified diff to a file, run "
        f"`python3 {_DIFF_PNG_TOOL} <diff file> <out prefix>` (it prints one PNG "
        "path per line) and @@attach each. "
        "The user can pick the model for this thread or the whole channel in plain "
        "words ('use Fable for this thread', 'make Haiku this channel's default', "
        "'back to the default model'). Then reply with ONLY a marker line, "
        "'@@model thread <model>' or '@@model channel <model>' (the model 'default' "
        "clears a pick), plus at most one short sentence, and don't start the task: "
        "the bridge switches the model, drops the marker and re-sends the message. "
        "<model> is opus, sonnet, haiku, fable, best or opusplan (optionally with a "
        "[1m] suffix), or a full claude-... id. Asked for the model this session "
        "already runs on? Nothing to switch: just do the task."
    )

    # `claude -p` exits when the turn ends and takes every process it started
    # with it — background shell jobs and sub-agents too — so a turn must
    # outlive all the work it launched.
    _TURN_LIFETIME_RULE = (
        "Nothing outlives your turn: when it ends, this process exits and every "
        "process it started dies with it, background shell jobs and sub-agents "
        "included. Never end your turn while work you launched is still running: "
        "wait on every agent and long job until it reports finished (run agents in "
        "the foreground, or block on background ones; if a wait times out while the "
        "job still runs, wait again), then reply."
    )

    @staticmethod
    def _build_cmd(
        session_id: str | None = None,
        resume: str | None = None,
        plugin_dir: str | None = None,
        model: str | None = None,
    ) -> list[str]:
        # stream-json + --verbose makes the CLI emit one event per line on
        # stdout (system/init, assistant text, thinking, tool_use,
        # tool_result, result). We log each event as it arrives so Docker
        # captures Claude's full trace, not just the final reply.
        # The prompt names the session's model so Claude can tell a no-op
        # "use <current model>" from a real switch.
        system_prompt = (
            ClaudeHandler._FLOW_B_SYSTEM_PROMPT
            + " " + ClaudeHandler._TURN_LIFETIME_RULE
            + f" This session currently runs on: {model or 'the default model'}."
        )
        cmd = [
            "claude", "-p",
            "--dangerously-skip-permissions",
            "--append-system-prompt", system_prompt,
            "--output-format", "stream-json",
            "--verbose",
        ]
        if model:
            cmd.extend(["--model", model])
        if plugin_dir:
            cmd.extend(["--plugin-dir", plugin_dir])
        if session_id:
            cmd.extend(["--session-id", session_id])
        if resume:
            cmd.extend(["--resume", resume])
        return cmd

    async def _run_claude(
        self, cmd: list[str], prompt: str, cwd: str | None = None,
        thread_ts: str | None = None, channel: str | None = None,
        progress_cb: ProgressCb | None = None,
    ) -> str:
        """Spawn a ``claude -p`` subprocess, stream-log its events, and return the final reply.

        There is no wall-clock cap: a run lives until it finishes, the user 🛑s
        it, or the idle watchdog kills it after IDLE_TIMEOUT seconds with no
        stream activity. While it runs, *progress_cb* (if given) is invoked with
        an aggregated :class:`_Progress` snapshot at most once per
        PROGRESS_INTERVAL, then once more with ``done=True`` for the summary.
        """
        env = os.environ.copy()
        # Strip tokens that must never be reachable by the Claude subprocess.
        # A prompt-injection attack could otherwise instruct Claude to exfiltrate them.
        for _key in ("CLAUDECODE", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "ANTHROPIC_API_KEY"):
            env.pop(_key, None)

        logger.debug("claude spawn: cwd=%s cmd=%s prompt=%r", cwd, cmd, prompt[:500])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                limit=STREAM_BUFFER_LIMIT,
                start_new_session=True,
            )
        except FileNotFoundError:
            # create_subprocess_exec raises FileNotFoundError for two unrelated
            # causes: the claude binary is not on PATH, or `cwd` (the project
            # dir from projects.json) does not exist. Don't blame the CLI for a
            # bad project path — name the path so the fix is obvious.
            if cwd is not None and not os.path.isdir(cwd):
                logger.error("Project directory does not exist: %s", cwd)
                return (
                    f"Sorry, the project directory doesn't exist: `{cwd}`. "
                    "Fix the path in projects.json and run ./reload-mapping.sh."
                )
            logger.error("claude CLI not found — is it installed and in PATH?")
            return "Sorry, the Claude CLI is not available."

        if thread_ts is not None:
            # One in-flight process per thread_ts. The daemon's _active_threads
            # guard prevents a second concurrent run for the same thread, so this
            # never overwrites a still-tracked process. If that guard is ever
            # relaxed, this map (and stop()) would need a per-run key instead.
            self._processes[thread_ts] = process
            session_store.upsert(
                thread_ts, in_flight=True, pid=process.pid,
                channel=channel, path=self._store_path,
            )

        def _mark_finished(ctx_tokens: int | None = None) -> None:
            # The subprocess has terminated (cleanly or with an error) — the run
            # is no longer in flight. Only a run that never reaches a terminal
            # path (the daemon itself dying mid-run) stays in_flight=true, which
            # is exactly what boot crash-recovery should surface.
            # ctx_tokens (clean finish only) feeds the model-switch context guard.
            if thread_ts is not None:
                if ctx_tokens:
                    self._thread_ctx[thread_ts] = ctx_tokens
                session_store.upsert(
                    thread_ts, in_flight=False, pid=None, ctx_tokens=ctx_tokens,
                    path=self._store_path,
                )

        try:
            # Send prompt and close stdin so claude can begin work.
            assert process.stdin is not None
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

            final_result: str | None = None
            loop = asyncio.get_running_loop()
            # Prior thread spend (unrounded) so the meta can show msg + thread total.
            prior_cost = 0.0
            if thread_ts is not None:
                rec = session_store.load(self._store_path).get(thread_ts) or {}
                prior_cost = rec.get("cum_cost") or 0.0
            tracker = _ProgressTracker(loop.time(), prior_cost=prior_cost)
            last_event = loop.time()
            flushed_once = False
            idle_killed = False
            # Recent stderr + non-JSON stdout + the last result event feed error
            # classification: a usage/quota limit often arrives with EMPTY stderr,
            # as a result event or a plain stdout line, so stderr alone misses it.
            stderr_tail: deque[str] = deque(maxlen=25)
            stdout_tail: deque[str] = deque(maxlen=10)
            result_meta: dict | None = None

            def _error_signals() -> list[str]:
                """All the text a failed run can carry, for _classify_run_error."""
                return list(stderr_tail) + list(stdout_tail) + [_result_error_text(result_meta)]

            async def consume_stdout() -> None:
                nonlocal final_result, last_event, result_meta
                assert process.stdout is not None
                async for raw_line in process.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        continue
                    last_event = loop.time()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        stdout_tail.append(line)  # banner / plain-text error line
                        logger.debug("claude stdout (non-json): %s", line[:1000])
                        continue
                    self._log_stream_event(event)
                    tracker.ingest(event)
                    if isinstance(event, dict) and event.get("type") == "result":
                        result_meta = event
                        if "result" in event:
                            final_result = event["result"]

            async def consume_stderr() -> None:
                nonlocal last_event
                assert process.stderr is not None
                async for raw_line in process.stderr:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    if line:
                        last_event = loop.time()
                        stderr_tail.append(line)
                        logger.warning("claude stderr: %s", line[:1000])

            async def monitor() -> None:
                # Wakes every PROGRESS_INTERVAL to (1) kill genuinely-stuck runs
                # (no activity for IDLE_TIMEOUT) and (2) push one aggregated
                # progress update covering everything since the last push.
                nonlocal flushed_once, idle_killed
                interval = PROGRESS_INTERVAL if PROGRESS_INTERVAL > 0 else 10.0
                while True:
                    await asyncio.sleep(interval)
                    now = loop.time()
                    if IDLE_TIMEOUT > 0 and (now - last_event) >= IDLE_TIMEOUT:
                        idle_killed = True
                        logger.error(
                            "Claude run idle for %.0fs (>=%ss) — killing as stuck.",
                            now - last_event, IDLE_TIMEOUT,
                        )
                        _kill_process_tree(process)
                        return
                    idle = now - last_event
                    # Hold the FIRST update until there's something to show — an
                    # empty "🔄 Working… · 0 tools" panel looks broken. Once we've
                    # posted, keep flushing every tick so elapsed/idle stay live.
                    if progress_cb is not None and (flushed_once or tracker.has_content()):
                        try:
                            tracker._usage = await quota.get(now)  # 5h/7d only; cached, fails open
                            await progress_cb(tracker.snapshot(now, idle=idle))
                            flushed_once = True
                        except Exception as exc:  # noqa: BLE001 — progress is best-effort
                            logger.warning("progress update failed: %s", exc)

            stdout_task = asyncio.create_task(consume_stdout())
            stderr_task = asyncio.create_task(consume_stderr())
            monitor_task = asyncio.create_task(monitor())

            try:
                await asyncio.gather(stdout_task, stderr_task, process.wait())
            finally:
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task

            if idle_killed:
                _mark_finished()
                return _TIMEOUT_REPLY

            if process.returncode != 0:
                logger.error(
                    "Claude CLI failed (rc=%d) cmd=%s prompt=%r",
                    process.returncode, cmd, prompt[:200],
                )
                _mark_finished()
                if process.returncode < 0:
                    # Killed by a signal (deliberate stop / OOM), not a transient
                    # error — do not let the resume policy retry it.
                    return _INTERRUPTED_REPLY
                # Surface WHY when a known cause shows up (usage/quota limit, rate
                # limit, server error, …) instead of the bare generic apology.
                reason = _classify_run_error(_error_signals())
                if reason:
                    return f"{_RUN_FAILURE_PREFIX} ({reason})"
                return _RUN_FAILURE_PREFIX

            if final_result is None:
                logger.warning("Claude stream ended with no result event.")
                _mark_finished()
                # An error result with no text (e.g. a quota limit) lands here, not
                # in the rc!=0 branch — classify the same signals before giving up.
                reason = _classify_run_error(_error_signals())
                if reason:
                    return f"{_RUN_FAILURE_PREFIX} ({reason})"
                return "Sorry, I couldn't parse the response."
            # Always emit the done snapshot: a posted live status collapses to the
            # summary; a fast run that never posted still hands its meta to the
            # reporter for the reply's meta line (the reporter won't post a new
            # status message in that case — see _ProgressReporter.__call__).
            if progress_cb is not None:
                with contextlib.suppress(Exception):
                    tracker._usage = await quota.get(loop.time(), final=True)  # + refresh Fable
                    await progress_cb(tracker.snapshot(loop.time(), done=True))
            _mark_finished(tracker.ctx_last or None)
            if thread_ts is not None:
                pct = tracker._ctx_pct()  # remember for next turn's nudge decision
                fields: dict[str, Any] = {}
                if pct is not None:
                    fields["ctx_pct"] = pct
                run_cost = tracker.run_cost()
                if run_cost is not None:  # grow the unrounded thread total by this run
                    fields["cum_cost"] = prior_cost + run_cost
                if fields:
                    session_store.upsert(thread_ts, path=self._store_path, **fields)
            return final_result
        finally:
            if thread_ts is not None:
                self._processes.pop(thread_ts, None)

    @staticmethod
    def _log_stream_event(event: Any) -> None:
        """Log a single stream-json event from ``claude -p`` in human-readable form.

        All per-event logs are at DEBUG so the default INFO level matches the
        pre-stream-json behaviour (lifecycle only). Set ``LOG_LEVEL=DEBUG`` to
        see the full trace of Claude's tool calls and reasoning.
        """
        if not isinstance(event, dict):
            return
        etype = event.get("type")
        if etype == "system":
            logger.debug(
                "claude stream: system/%s session=%s cwd=%s tools=%s",
                event.get("subtype", ""),
                event.get("session_id", ""),
                event.get("cwd", ""),
                event.get("tools", ""),
            )
        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        logger.debug("claude text: %s", text[:2000])
                elif btype == "thinking":
                    thought = (block.get("thinking") or "").strip()
                    if thought:
                        logger.debug("claude thinking: %s", thought[:2000])
                elif btype == "tool_use":
                    logger.debug(
                        "claude tool_use: %s id=%s input=%s",
                        block.get("name", ""),
                        block.get("id", ""),
                        json.dumps(block.get("input", {}), ensure_ascii=False)[:2000],
                    )
        elif etype == "user":
            for block in event.get("message", {}).get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                logger.debug(
                    "claude tool_result%s id=%s: %s",
                    " (error)" if block.get("is_error") else "",
                    block.get("tool_use_id", ""),
                    str(content)[:2000],
                )
        elif etype == "result":
            logger.debug(
                "claude stream: result subtype=%s duration_ms=%s num_turns=%s usage=%s",
                event.get("subtype", ""),
                event.get("duration_ms", ""),
                event.get("num_turns", ""),
                event.get("usage", ""),
            )
        else:
            logger.debug("claude stream: %s %s", etype, json.dumps(event, ensure_ascii=False)[:500])

    async def _build_thread_prompt(self, channel: str, thread_ts: str) -> str:
        """Fetch Slack thread history and format as a conversation prompt."""
        resp = await self._slack_client.conversations_replies(
            channel=channel, ts=thread_ts
        )
        messages = resp.get("messages", [])

        lines = ["The following is a Slack conversation. Continue assisting the user.\n"]
        for msg in messages:
            is_bot = (
                msg.get("user") == self._bot_user_id
                or msg.get("bot_id")
            )
            label = "[Assistant]" if is_bot else "[Human]"
            text = msg.get("text", "")
            lines.append(f"{label}: {text}")

        return "\n".join(lines)
