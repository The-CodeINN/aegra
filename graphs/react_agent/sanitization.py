"""Unicode and tool-input sanitization utilities.

Ported from Claude-code's ``sanitization.ts`` — strips invisible/directional
Unicode characters that could be used for prompt injection, homograph attacks,
or bi-directional text spoofing.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode stripping
# ---------------------------------------------------------------------------

# Characters used in common text-injection / prompt-hijacking attacks:
#   U+200B–U+200F  zero-width space / non-joiner / joiner / LRM / RLM
#   U+202A–U+202E  bi-directional embedding / override controls
#   U+2066–U+2069  directional isolate / embed controls
#   U+FEFF         BOM / zero-width no-break space
#   U+E000–U+F8FF  private-use area (PUA)
_STRIP_PATTERN = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\ue000-\uf8ff]",
    re.UNICODE,
)

# Maximum iterations for the normalization loop — prevents pathological inputs
# from looping forever (e.g., strings whose NFC form re-introduces stripped chars)
_MAX_NORMALIZE_ROUNDS = 10


def sanitize_unicode(text: str) -> str:
    """Normalize *text* to NFKC and strip dangerous Unicode control characters.

    Applies up to ``_MAX_NORMALIZE_ROUNDS`` rounds of NFKC normalization
    followed by stripping of invisible/directional characters until the output
    stabilizes.  This mirrors the multi-pass approach in Claude-code's
    ``sanitizeText`` to handle encodings that expand into more control chars
    after normalization.

    Args:
        text: The raw input string (e.g. a user message or tool argument).

    Returns:
        The sanitized string with dangerous characters removed.
    """
    for _ in range(_MAX_NORMALIZE_ROUNDS):
        normalized = unicodedata.normalize("NFKC", text)
        stripped = _STRIP_PATTERN.sub("", normalized)
        if stripped == text:
            return stripped
        text = stripped
    return text


# ---------------------------------------------------------------------------
# Tool argument ID validator
# ---------------------------------------------------------------------------

# Mongo ObjectId / UUID / slug-style IDs used by the platform.
# Allows: alphanumeric, underscore, hyphen; 1–64 chars.
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def validate_resource_id(value: str, field_name: str = "id") -> str:
    """Validate that *value* is a safe resource identifier.

    Raises ``ValueError`` if *value* contains characters outside the
    expected alphanumeric/underscore/hyphen set or exceeds 64 characters.
    Returns the value unchanged when valid.

    This defence-in-depth check prevents crafted tool arguments from
    injecting shell meta-characters or oversized strings into downstream
    MongoDB / HTTP calls.
    """
    if not _ID_PATTERN.match(value):
        raise ValueError(f"Invalid {field_name!r}: must be 1-64 alphanumeric/underscore/hyphen characters")
    return value
