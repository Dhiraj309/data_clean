"""LaughLM custom text filters.

Only filters that are intentionally *not* provided by DataTrove live here.
The functions are pure Python and configured from the per-dataset Stage-2 YAML.

Important design choices:
- No generic reference/citation stripping. Citations are useful training data.
- Horizontal whitespace is preserved so code, Markdown tables, math and other
  preformatted material are not damaged.
- Heavy ML quality filtering is not repeated on already-curated HF datasets.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

FilterResult = Tuple[Optional[str], Optional[str]]
FilterFn = Callable[..., FilterResult]

FILTER_REGISTRY: Dict[str, FilterFn] = {}


def register_filter(name: str) -> Callable[[FilterFn], FilterFn]:
    """Register a custom filter under a stable config name."""

    def _decorator(fn: FilterFn) -> FilterFn:
        if name in FILTER_REGISTRY:
            raise ValueError(f"A filter named '{name}' is already registered.")
        FILTER_REGISTRY[name] = fn
        return fn

    return _decorator


def load_plugins(paths: List[str]) -> None:
    """Load optional user filter modules."""
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Plugin path does not exist: {path}")
        module_name = f"filters_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


@register_filter("normalize_whitespace")
def normalize_whitespace(
    text: str,
    max_blank_lines: int = 2,
    strip: bool = True,
    **_: object,
) -> FilterResult:
    """Normalize line endings and excessive blank lines only.

    Deliberately does *not* collapse spaces/tabs. Doing so damages Python/code
    indentation, Markdown tables, ASCII diagrams and some mathematical text.
    """
    if not text:
        return None, "empty_input"
    if max_blank_lines < 0:
        raise ValueError("max_blank_lines must be >= 0")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # N blank lines correspond to N+1 consecutive newline characters.
    max_newlines = max_blank_lines + 1
    text = re.sub(rf"\n{{{max_newlines + 1},}}", "\n" * max_newlines, text)
    if strip:
        text = text.strip()

    if not text:
        return None, "empty_after_whitespace_normalization"
    return text, None


@register_filter("length_filter")
def length_filter(
    text: str,
    min_chars: int = 1,
    max_chars: int = 10_000_000,
    **_: object,
) -> FilterResult:
    """Reject catastrophic character-count outliers."""
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("Invalid min_chars/max_chars bounds")
    n_chars = len(text)
    if n_chars < min_chars:
        return None, "too_short_chars"
    if n_chars > max_chars:
        return None, "too_long_chars"
    return text, None


_toxicity_wordlist_cache: Dict[str, set[str]] = {}


@register_filter("toxicity_wordlist_filter")
def toxicity_wordlist_filter(
    text: str,
    banned_terms_path: Optional[str] = None,
    max_banned_term_hits: int = 0,
    case_sensitive: bool = False,
    **_: object,
) -> FilterResult:
    """Optional conservative word-list rejection hook.

    This is disabled by default. A word list alone is a crude toxicity signal,
    so it should only be enabled after measuring false-positive rates on a
    representative sample of each domain.
    """
    if not banned_terms_path:
        return text, None
    if max_banned_term_hits < 0:
        raise ValueError("max_banned_term_hits must be >= 0")

    cache_key = f"{Path(banned_terms_path).resolve()}::{case_sensitive}"
    if cache_key not in _toxicity_wordlist_cache:
        path = Path(banned_terms_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"banned_terms_path not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            terms = {
                (line.strip() if case_sensitive else line.strip().lower())
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
        _toxicity_wordlist_cache[cache_key] = terms

    probe = text if case_sensitive else text.lower()
    tokens = re.findall(r"[\w'-]+", probe, flags=re.UNICODE)
    terms = _toxicity_wordlist_cache[cache_key]
    hits = sum(1 for token in tokens if token in terms)
    if hits > max_banned_term_hits:
        return None, "toxicity_wordlist_threshold_exceeded"
    return text, None


def run_pipeline(text: str, steps: List[Dict[str, object]]) -> FilterResult:
    """Run custom filters in config order and stop on first rejection."""
    current: Optional[str] = text
    for step in steps:
        if current is None:
            return None, "unexpected_none"
        name = str(step["name"])
        kwargs = step.get("kwargs", {}) or {}
        if not isinstance(kwargs, dict):
            raise TypeError(f"Filter '{name}' kwargs must be a mapping")
        fn = FILTER_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown filter '{name}'. Registered: {sorted(FILTER_REGISTRY)}")
        current, reason = fn(current, **kwargs)
        if current is None:
            return None, reason
    return current, None

# High-confidence credential patterns. This is intentionally conservative; it
# is meant as a safety net for code corpora, not a general entropy scanner.
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,255}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
]


@register_filter("secret_filter")
def secret_filter(
    text: str,
    max_secret_hits: int = 0,
    **_: object,
) -> FilterResult:
    """Reject documents containing high-confidence credential material."""
    if max_secret_hits < 0:
        raise ValueError("max_secret_hits must be >= 0")
    hits = sum(len(pattern.findall(text)) for pattern in _SECRET_PATTERNS)
    if hits > max_secret_hits:
        return None, "secret_material_detected"
    return text, None
