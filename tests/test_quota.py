"""Unit tests for src/quota.py — two-source quota (unified headers + scoped usage)."""

import asyncio

import quota

_NOW = 1_000_000.0


def _icon(p):
    return f"[{p}]"


def _snap():
    return {
        "windows": [
            {"label": "5h", "pct": 3, "reset": int(_NOW + 2 * 3600)},
            {"label": "7d", "pct": 57, "reset": int(_NOW + 4 * 86400)},
        ],
        "scoped": {"name": "Fable", "pct": 70},
    }


def test_format_windows_and_scoped():
    assert quota.format_status(_snap(), _icon, now=_NOW) == \
        "[3] 5h 3% ⟳2h · [57] 7d 57% ⟳4d · [70] Fable 70%"


def test_format_scoped_absent_and_no_reset():
    snap = {"windows": [{"label": "5h", "pct": 3, "reset": None}], "scoped": None}
    assert quota.format_status(snap, _icon, now=_NOW) == "[3] 5h 3%"


def test_countdown_units_and_fail_open():
    assert quota._countdown(int(_NOW + 1800), _NOW) == "30m"
    assert quota._countdown(int(_NOW + 3 * 3600), _NOW) == "3h"
    assert quota._countdown(int(_NOW + 4 * 86400), _NOW) == "4d"
    assert quota._countdown(int(_NOW - 5), _NOW) == "now"
    assert quota._countdown(None, _NOW) == ""
    for bad in (None, {}, "nope"):
        assert quota.format_status(bad, _icon, now=_NOW) is None


def test_parse_unified_headers():
    h = {
        "anthropic-ratelimit-unified-5h-utilization": "0.03",
        "anthropic-ratelimit-unified-5h-reset": "1787262000",
        "anthropic-ratelimit-unified-7d-utilization": "0.57",
        "anthropic-ratelimit-unified-7d-reset": "1787580000",
    }
    assert quota._parse_unified(h) == [
        {"label": "5h", "pct": 3, "reset": 1787262000},
        {"label": "7d", "pct": 57, "reset": 1787580000},
    ]
    assert quota._parse_unified({}) is None


def test_parse_scoped():
    good = {"limits": [
        {"kind": "weekly_all", "percent": 57},
        {"kind": "weekly_scoped", "is_active": False, "percent": 9, "scope": {}},
        {"kind": "weekly_scoped", "is_active": True, "percent": 70,
         "scope": {"model": {"display_name": "Fable"}}},
    ]}
    assert quota._parse_scoped(good) == {"name": "Fable", "pct": 70}
    # Regression: Fable below the overall weekly % reads is_active=False — still shown.
    assert quota._parse_scoped({"limits": [
        {"kind": "weekly_all", "percent": 25, "is_active": True},
        {"kind": "weekly_scoped", "is_active": False, "percent": 23,
         "scope": {"model": {"display_name": "Fable"}}},
    ]}) == {"name": "Fable", "pct": 23}
    assert quota._parse_scoped({"limits": [{"kind": "weekly_all", "percent": 5}]}) is None
    assert quota._parse_scoped(None) is None


def _fresh_cache(monkeypatch, ttl=60.0):
    monkeypatch.setattr(quota, "_TTL", ttl)
    monkeypatch.setattr(quota, "_cache", {
        "unified_at": None, "unified": None, "unified_retry_at": None,
        "scoped_at": None, "scoped": None, "scoped_retry_at": None,
    })


def test_scoped_only_fetched_on_final(monkeypatch):
    uni = {"n": 0}
    sco = {"n": 0}
    monkeypatch.setattr(quota, "_fetch_unified",
                        lambda: (uni.__setitem__("n", uni["n"] + 1) or ([{"label": "5h", "pct": 1, "reset": None}], None)))
    monkeypatch.setattr(quota, "_fetch_scoped",
                        lambda: (sco.__setitem__("n", sco["n"] + 1) or ({"name": "Fable", "pct": 70}, None)))
    _fresh_cache(monkeypatch)

    async def run():
        a = await quota.get(0.0)                 # non-final → unified only
        b = await quota.get(100.0, final=True)   # TTL passed → unified + scoped
        return a, b

    a, b = asyncio.run(run())
    assert uni["n"] == 2 and sco["n"] == 1       # scoped fetched only on the final call
    assert a["scoped"] is None                   # not present before final
    assert b["scoped"] == {"name": "Fable", "pct": 70}


def test_scoped_cleared_when_bucket_disappears(monkeypatch):
    # Regression: Fable hit 100%, then the weekly_scoped bucket disappeared.
    # A successful fetch that parses to None must clear the cache, not freeze the stale %.
    result = {"v": ({"name": "Fable", "pct": 100}, None)}
    monkeypatch.setattr(quota, "_fetch_unified", lambda: ([{"label": "5h", "pct": 1, "reset": None}], None))
    monkeypatch.setattr(quota, "_fetch_scoped", lambda: result["v"])
    _fresh_cache(monkeypatch)

    async def run():
        a = await quota.get(0.0, final=True)                 # bucket active → cache 100%
        result["v"] = (quota._EMPTY, None)                   # server ok, bucket gone
        b = await quota.get(100.0, final=True)               # TTL passed → refetch, clears
        return a, b

    a, b = asyncio.run(run())
    assert a["scoped"] == {"name": "Fable", "pct": 100}
    assert b["scoped"] is None                                # no longer frozen


def test_unified_429_backoff(monkeypatch):
    calls = {"n": 0}

    def fake_unified():
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"label": "5h", "pct": 3, "reset": None}], None
        return None, 1800.0                      # 429

    monkeypatch.setattr(quota, "_fetch_unified", fake_unified)
    monkeypatch.setattr(quota, "_fetch_scoped", lambda: (None, None))
    _fresh_cache(monkeypatch)

    async def run():
        await quota.get(0.0)                      # fetch #1 ok → cache 3%
        await quota.get(100.0)                    # TTL passed → fetch #2 → 429, backoff to 1900
        blocked = await quota.get(500.0)          # within backoff → NO fetch, serves stale
        await quota.get(2000.0)                    # past retry_at → fetch #3
        return blocked

    blocked = asyncio.run(run())
    assert calls["n"] == 3                        # the 500.0 call did not fetch
    assert blocked["windows"][0]["pct"] == 3     # stale value kept through 429


def test_get_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(quota, "_TTL", 0.0)
    assert asyncio.run(quota.get(1.0)) is None
