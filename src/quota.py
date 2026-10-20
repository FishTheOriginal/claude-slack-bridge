"""Anthropic plan-quota (5-hour / weekly) lookup for the live status line.

Two sources, because no single one gives everything cheaply:

* **5h + 7d** come from the ``anthropic-ratelimit-unified-*`` response headers on
  a **normal ``/v1/messages`` call** — the same numbers the official meters read.
  We make a minimal probe (``max_tokens:1``, ~1 output token) whose headers carry
  the account-wide 5h/7d utilization + reset. The messages endpoint has your full
  plan limit, so this never hits the special low-limit lockout that
  ``/api/oauth/usage`` does.

* **The model-scoped bucket (e.g. Fable)** only exists in ``/api/oauth/usage``,
  which rate-limits hard (HTTP 429, ``Retry-After`` ~30 min). So we refresh it
  **only on a run's final snapshot**, cache it between, and honor ``Retry-After``.

Everything fails open: any error keeps the last good value (or ``None``) and the
caller simply omits the segment. Refreshes are gated by ``QUOTA_TTL`` seconds
(default 300; ≤0 disables the feature).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request

_USAGE_ENDPOINT = "https://api.anthropic.com/api/oauth/usage"      # model-scoped bucket only
_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"       # carries unified 5h/7d headers
_OAUTH_BETA = "oauth-2025-04-20"
_API_VERSION = "2023-06-01"
_PROBE_MODEL = os.environ.get("QUOTA_PROBE_MODEL", "claude-haiku-4-5-20251001")
_TTL = float(os.environ.get("QUOTA_TTL", "300"))  # seconds; ≤0 disables the feature
_HTTP_TIMEOUT = 4.0                                # network cap (runs in a worker thread)
_BACKOFF_FALLBACK = 1800.0                         # 429 with no/bad Retry-After → wait 30 min

# A fetch that reached the server but found nothing (no unified headers, or no
# model-scoped bucket) returns this — distinct from ``None`` (fetch failed),
# so ``get`` clears the stale value instead of serving it forever.
_EMPTY = object()

_lock = asyncio.Lock()
# ``*_at``: monotonic time (loop.time()) of last attempt. ``*_retry_at``: while set
# and in the future, that source is in 429 backoff and is skipped. ``unified`` is a
# list of window dicts {label,pct,reset}; ``scoped`` is {name,pct} or None.
_cache: dict = {
    "unified_at": None, "unified": None, "unified_retry_at": None,
    "scoped_at": None, "scoped": None, "scoped_retry_at": None,
}


def _creds_path() -> str:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return os.path.join(base, ".credentials.json")


def _token() -> "str | None":
    try:
        with open(_creds_path()) as fh:
            return json.load(fh)["claudeAiOauth"]["accessToken"]
    except (OSError, KeyError, ValueError):
        return None


def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        return float(e.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return _BACKOFF_FALLBACK


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-beta": _OAUTH_BETA,
        "anthropic-version": _API_VERSION,
        "content-type": "application/json",
    }


def _parse_unified(h) -> "list | None":
    """Pull 5h/7d windows from unified rate-limit response headers. None if absent."""
    windows = []
    for label, prefix in (("5h", "5h"), ("7d", "7d")):
        util = h.get(f"anthropic-ratelimit-unified-{prefix}-utilization")
        if util is None:
            continue
        try:
            pct = round(float(util) * 100)
        except (TypeError, ValueError):
            continue
        try:
            reset = int(h.get(f"anthropic-ratelimit-unified-{prefix}-reset"))
        except (TypeError, ValueError):
            reset = None
        windows.append({"label": label, "pct": pct, "reset": reset})
    return windows or None


def _fetch_unified() -> "tuple[list | None, float | None]":
    """Minimal /v1/messages probe; read 5h/7d from its headers (~1 output token).

    Returns ``(windows, backoff)``: ``(list, None)`` ok; ``(_EMPTY, None)`` when the
    fetch succeeded but no windows were present; ``(None, seconds)`` on 429;
    ``(None, None)`` on any other failure.
    """
    token = _token()
    if not token:
        return None, None
    body = json.dumps({
        "model": _PROBE_MODEL, "max_tokens": 1,
        "messages": [{"role": "user", "content": "."}],
    }).encode()
    req = urllib.request.Request(_MESSAGES_ENDPOINT, data=body, headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return _parse_unified(resp.headers) or _EMPTY, None
    except urllib.error.HTTPError as e:
        return (None, _retry_after(e)) if e.code == 429 else (None, None)
    except Exception:  # noqa: BLE001 — best-effort; never break the status line
        return None, None


def _parse_scoped(data) -> "dict | None":
    """Model-scoped weekly bucket (e.g. Fable) from a usage payload; highest % if several.

    ``is_active`` is ignored: it only flags the limit that's currently binding, so a
    Fable bucket below the overall weekly % reads inactive yet is still real."""
    if not isinstance(data, dict):
        return None
    scoped = [l for l in data.get("limits") or []
              if isinstance(l, dict) and l.get("kind") == "weekly_scoped"]
    if not scoped:
        return None
    lim = max(scoped, key=lambda l: l.get("percent") or 0)
    name = (((lim.get("scope") or {}).get("model") or {}).get("display_name")) or "scoped"
    return {"name": name, "pct": round(lim.get("percent") or 0)}


def _fetch_scoped() -> "tuple[dict | None, float | None]":
    """GET /api/oauth/usage for the model-scoped bucket. Same return shape as _fetch_unified
    (``_EMPTY`` when the bucket is gone, so a stale value is cleared, not frozen)."""
    token = _token()
    if not token:
        return None, None
    req = urllib.request.Request(_USAGE_ENDPOINT, headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return _parse_scoped(json.load(resp)) or _EMPTY, None
    except urllib.error.HTTPError as e:
        return (None, _retry_after(e)) if e.code == 429 else (None, None)
    except Exception:  # noqa: BLE001
        return None, None


async def get(now: float, *, final: bool = False) -> "dict | None":
    """Return the merged quota snapshot ``{"windows": [...], "scoped": {...}}``.

    *now* is monotonic seconds (``loop.time()``), used only for TTL/backoff. The 5h/7d
    windows refresh every ``_TTL``; the model-scoped bucket refreshes only when
    *final* is true (a run's last snapshot) — cached between. Each source has its own
    429 backoff. Fetches run in a worker thread so the event loop never blocks.
    """
    if _TTL <= 0:
        return None
    async with _lock:
        if _cache["unified_retry_at"] is None or now >= _cache["unified_retry_at"]:
            if _cache["unified_at"] is None or (now - _cache["unified_at"]) >= _TTL:
                data, backoff = await asyncio.to_thread(_fetch_unified)
                if data is not None:                      # definitive answer (windows or _EMPTY)
                    _cache["unified"] = None if data is _EMPTY else data
                    _cache["unified_retry_at"] = None
                elif backoff is not None:
                    _cache["unified_retry_at"] = now + backoff
                _cache["unified_at"] = now
        if final and (_cache["scoped_retry_at"] is None or now >= _cache["scoped_retry_at"]):
            if _cache["scoped_at"] is None or (now - _cache["scoped_at"]) >= _TTL:
                data, backoff = await asyncio.to_thread(_fetch_scoped)
                if data is not None:                      # definitive answer (bucket or _EMPTY)
                    _cache["scoped"] = None if data is _EMPTY else data
                    _cache["scoped_retry_at"] = None
                elif backoff is not None:
                    _cache["scoped_retry_at"] = now + backoff
                _cache["scoped_at"] = now
        return {"windows": _cache["unified"], "scoped": _cache["scoped"]}


def _countdown(reset: "int | None", now: float) -> str:
    """Compact time-until-reset for an absolute epoch-seconds *reset*: ``40m`` /
    ``3h`` / ``4d``. ``""`` when missing, ``now`` if already past."""
    if not reset:
        return ""
    secs = reset - now
    if secs <= 0:
        return "now"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        return f"{int(secs // 3600)}h"
    return f"{int(secs // 86400)}d"


def format_status(snap: "dict | None", icon, now: "float | None" = None) -> "str | None":
    """Render the quota segment, e.g. ``🟢 5h 3% ⟳2h · 🟠 7d 57% ⟳4d · 🔴 Fable 70%``.

    *snap* is what ``get`` returns. ``icon`` maps a percent → glyph; ``⟳`` marks the
    compact time until each window resets. *now* is epoch seconds (defaults to wall
    clock). Returns ``None`` when nothing usable is present.
    """
    if not isinstance(snap, dict):
        return None
    if now is None:
        now = time.time()
    segs: list[str] = []
    for w in snap.get("windows") or []:
        pct = w.get("pct")
        if pct is None:
            continue
        r = _countdown(w.get("reset"), now)
        segs.append(f"{icon(pct)} {w.get('label', '?')} {pct}%" + (f" ⟳{r}" if r else ""))
    scoped = snap.get("scoped")
    if isinstance(scoped, dict) and scoped.get("pct") is not None:
        p = scoped["pct"]
        segs.append(f"{icon(p)} {scoped.get('name', 'scoped')} {p}%")
    return " · ".join(segs) or None


def demo() -> None:
    """Self-check for the pure formatters. Run: ``python quota.py``."""
    now = 1_000_000.0
    snap = {
        "windows": [
            {"label": "5h", "pct": 3, "reset": int(now + 2 * 3600)},    # +2h
            {"label": "7d", "pct": 57, "reset": int(now + 4 * 86400)},  # +4d
        ],
        "scoped": {"name": "Fable", "pct": 70},
    }
    out = format_status(snap, lambda p: f"[{p}]", now=now)
    assert out == "[3] 5h 3% ⟳2h · [57] 7d 57% ⟳4d · [70] Fable 70%", out
    # scoped absent → just windows
    assert format_status({"windows": [{"label": "5h", "pct": 3, "reset": None}], "scoped": None},
                         lambda p: "•", now=now) == "• 5h 3%"
    assert _countdown(int(now + 1800), now) == "30m"
    assert _countdown(int(now - 5), now) == "now" and _countdown(None, now) == ""
    # parsers
    assert _parse_unified({
        "anthropic-ratelimit-unified-5h-utilization": "0.03",
        "anthropic-ratelimit-unified-5h-reset": "1787262000",
        "anthropic-ratelimit-unified-7d-utilization": "0.57",
        "anthropic-ratelimit-unified-7d-reset": "1787580000",
    }) == [{"label": "5h", "pct": 3, "reset": 1787262000},
           {"label": "7d", "pct": 57, "reset": 1787580000}]
    assert _parse_unified({}) is None
    assert _parse_scoped({"limits": [
        {"kind": "weekly_scoped", "is_active": True, "percent": 70,
         "scope": {"model": {"display_name": "Fable"}}}]}) == {"name": "Fable", "pct": 70}
    assert _parse_scoped({"limits": [{"kind": "weekly_all", "percent": 5}]}) is None
    # fail open
    for bad in (None, {}, "nope"):
        assert format_status(bad, lambda p: "•", now=now) is None
    print("ok")


if __name__ == "__main__":
    demo()
