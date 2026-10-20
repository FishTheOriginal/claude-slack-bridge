"""
security.py — User and channel-based access control for Slack interactions.

Supports user allowlists, channel allowlists, admin bypass, flexible/strict
modes, structured rejection logging, and a configurable rejection message.

All settings are loaded from environment variables so no code changes are
needed to enable or reconfigure access control at runtime.

Environment variables
---------------------
SECURITY_ENABLED            Enable access control; default "false".
SECURITY_STRICT_MODE        Deny when a relevant allowlist is empty; default "false".
SECURITY_ALLOWED_USERS      Comma-separated Slack user IDs that may use the bot.
SECURITY_ALLOWED_CHANNELS   Comma-separated Slack channel IDs that the bot responds in.
SECURITY_ADMIN_USERS        Comma-separated user IDs that bypass channel restrictions.
SECURITY_REJECTION_MESSAGE  Message posted to unauthorized users.
SECURITY_LOG_UNAUTHORIZED   Emit a warning log on each denial; default "true".
"""

import logging
import math
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes"}
_FALSY = {"0", "false", "no"}


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default


def _parse_id_set(value: str) -> set[str]:
    """Split a comma-separated string into a set of non-empty stripped tokens."""
    return {tok.strip() for tok in value.split(",") if tok.strip()}


@dataclass
class SecurityConfig:
    """Parsed access-control settings."""

    enabled: bool = False
    strict_mode: bool = False
    allowed_users: set[str] = field(default_factory=set)
    allowed_channels: set[str] = field(default_factory=set)
    admin_users: set[str] = field(default_factory=set)
    rejection_message: str = "You are not authorized to use this bot."
    log_unauthorized: bool = True

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Construct a SecurityConfig by reading environment variables."""
        cfg = cls(
            enabled=_parse_bool(os.environ.get("SECURITY_ENABLED", "false"), default=False),
            strict_mode=_parse_bool(os.environ.get("SECURITY_STRICT_MODE", "false"), default=False),
            allowed_users=_parse_id_set(os.environ.get("SECURITY_ALLOWED_USERS", "")),
            allowed_channels=_parse_id_set(os.environ.get("SECURITY_ALLOWED_CHANNELS", "")),
            admin_users=_parse_id_set(os.environ.get("SECURITY_ADMIN_USERS", "")),
            rejection_message=os.environ.get(
                "SECURITY_REJECTION_MESSAGE", "You are not authorized to use this bot."
            ),
            log_unauthorized=_parse_bool(
                os.environ.get("SECURITY_LOG_UNAUTHORIZED", "true"), default=True
            ),
        )
        if cfg.enabled:
            logger.info(
                "Access control enabled: strict=%s allowed_users=%d allowed_channels=%d admin_users=%d",
                cfg.strict_mode,
                len(cfg.allowed_users),
                len(cfg.allowed_channels),
                len(cfg.admin_users),
            )
        else:
            logger.info("Access control disabled (SECURITY_ENABLED not set).")
        return cfg


class AccessControl:
    """
    Enforces user and channel allowlists.

    Access logic
    ------------
    - Security disabled  → always allow.
    - Admin user         → passes channel check unconditionally; still subject to
                           the user allowlist (admins must be listed there too, or
                           the user allowlist must be empty).
    - Flexible mode      → a missing allowlist dimension is treated as "allow all"
                           for that dimension.
    - Strict mode        → a missing allowlist dimension is treated as "deny all"
                           for that dimension.
    """

    def __init__(self, config: SecurityConfig) -> None:
        self._cfg = config

    def is_allowed(self, user_id: str, channel_id: str) -> bool:
        """
        Return True if the user is permitted to interact in this channel.

        Args:
            user_id:    Slack user ID of the message author.
            channel_id: Slack channel ID where the message was posted.
        """
        cfg = self._cfg

        if not cfg.enabled:
            return True

        is_admin = user_id in cfg.admin_users

        # --- User allowlist check ---
        if cfg.allowed_users:
            if user_id not in cfg.allowed_users:
                self._deny(user_id, channel_id, "user_not_in_allowlist")
                return False
        elif cfg.strict_mode:
            self._deny(user_id, channel_id, "strict_mode_no_user_allowlist")
            return False

        # --- Channel allowlist check (admins are exempt) ---
        if not is_admin:
            if cfg.allowed_channels:
                if channel_id not in cfg.allowed_channels:
                    self._deny(user_id, channel_id, "channel_not_in_allowlist")
                    return False
            elif cfg.strict_mode:
                self._deny(user_id, channel_id, "strict_mode_no_channel_allowlist")
                return False

        logger.debug(
            "Access granted: user_id=%s channel_id=%s is_admin=%s",
            user_id, channel_id, is_admin,
        )
        return True

    def rejection_message(self) -> str:
        """Return the configured rejection message."""
        return self._cfg.rejection_message

    def _deny(self, user_id: str, channel_id: str, reason: str) -> None:
        if self._cfg.log_unauthorized:
            logger.warning(
                "Access denied: user_id=%s channel_id=%s reason=%s",
                user_id, channel_id, reason,
            )


# ---------------------------------------------------------------------------
# Secret redaction
#
# The live status ticker and error replies echo command lines, URLs and stderr,
# any of which can carry credentials. redact_secrets() masks the common shapes
# before that text is shown in Slack. It leans toward over-redaction on purpose:
# a hidden token is a cosmetic loss, a leaked one is a real one.
# ---------------------------------------------------------------------------

REDACTED = "[redacted]"

# Standalone provider key formats — masked whole, no surrounding context needed.
_SECRET_TOKENS: list[re.Pattern] = [
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{8,}"),          # Slack bot/user tokens
    re.compile(r"xapp-[A-Za-z0-9-]{8,}"),                # Slack app-level token
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{16,}"),       # OpenAI / Anthropic keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),           # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),         # GitHub fine-grained PAT
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),        # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),               # Google API key
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
]

# (pattern, group-number-holding-the-secret). Only that group is replaced, so the
# surrounding context (the key name, the flag, the header) stays readable. Order
# matters: bearer/basic runs before the header rule so the token (not just the
# scheme word) gets masked.
_SECRET_PAIRS: list[tuple[re.Pattern, int]] = [
    # bearer / basic credential after the scheme word.
    (re.compile(r"(?i)\b(?:bearer|basic)\s+([A-Za-z0-9._~+/=-]{16,})"), 1),
    # HTTP auth headers: Authorization: <value>, X-Api-Key: <value>.
    (re.compile(r"(?i)\b(?:authorization|proxy-authorization|x-api-key|x-auth-token)"
                r"\s*:\s*(\S+)"), 1),
    # key=value / key: value where the key names a credential.
    (re.compile(
        r"(?i)\b(?:pass(?:word|wd)?|secret|token|api[_-]?key|apikey|access[_-]?key"
        r"|auth(?:[_-]?token)?|client[_-]?secret|private[_-]?key|session[_-]?token)"
        r"\s*[=:]\s*('[^']*'|\"[^\"]*\"|\S+)"), 1),
    # Env-var names ending in a credential word (e.g. ANTHROPIC_API_KEY=,
    # GITHUB_TOKEN=, DB_PASSWORD=) — the leading "_" guards against words that
    # merely end in "key" (e.g. monkey). Masks the value regardless of entropy.
    (re.compile(r"(?i)\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*"
                r"_(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIALS?|APIKEY)"
                r"\s*[=:]\s*('[^']*'|\"[^\"]*\"|\S+)"), 1),
    # Long-form CLI credential flags: --password VALUE / --token=VALUE.
    (re.compile(r"(?i)--(?:password|pass|secret|token|api[_-]?key)(?:=|\s+)(\S+)"), 1),
]

# Generic unnamed tokens (e.g. ``T=ogk2r42…``) that no name/format rule covers.
# One principled axis instead of hand-tuned shape rules: an assignment value is
# masked only when it's both long enough and high-entropy (random-looking). Real
# tokens clear the bar; ``RUN=build-2026-06-30``, paths and words don't. Tunable
# via the two knobs below — raise them to redact less, lower to redact more.
_ENTROPY_MIN_LEN = 20         # values shorter than this are left alone
# Shannon bits/char. Measured separation: structured values (paths ~4.0, version
# strings ~4.0, dated build ids ~3.7) sit below this; random tokens (~4.7+) clear
# it. 4.2 keeps ``OUT=/home/…`` and ``v1.2.3-…`` readable while masking ``T=ogk2…``.
_ENTROPY_MIN_BITS = 4.2
_ASSIGN_RE = re.compile(r"(?<![\w/.-])([A-Za-z_]\w{0,40})=([^\s'\"]{%d,})" % _ENTROPY_MIN_LEN)


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of *s* in bits per character (0 for empty/uniform)."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mask_high_entropy_assign(m: "re.Match") -> str:
    """Mask ``NAME=value`` only when the value looks like a random token."""
    name, value = m.group(1), m.group(2)
    if _shannon_entropy(value) >= _ENTROPY_MIN_BITS:
        return f"{name}={REDACTED}"
    return m.group(0)


def redact_secrets(text: str) -> str:
    """Mask credential-looking substrings in *text* for safe display.

    Covers ``key=value`` / ``key: value`` credential pairs, HTTP auth headers,
    ``bearer``/``basic`` schemes, CLI password flags, high-entropy env values,
    and known provider key formats (Slack, OpenAI/Anthropic, GitHub, AWS,
    Google, JWT). Returns *text* unchanged when nothing matches.
    """
    if not text:
        return text
    for pat in _SECRET_TOKENS:
        text = pat.sub(REDACTED, text)
    for pat, grp in _SECRET_PAIRS:
        text = pat.sub(lambda m, g=grp: _mask_group(m, g), text)
    # Named/formatted secrets are masked above regardless of entropy; this last
    # pass catches the unnamed high-entropy leftovers without over-redacting.
    text = _ASSIGN_RE.sub(_mask_high_entropy_assign, text)
    return text


def _mask_group(m: "re.Match", group: int) -> str:
    """Replace only *group* of match *m* with REDACTED, keeping the rest."""
    whole = m.group(0)
    a = m.start(group) - m.start()
    b = m.end(group) - m.start()
    return whole[:a] + REDACTED + whole[b:]
