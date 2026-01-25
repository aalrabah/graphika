# response_parsing.py
from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """
    If model outputs ```json ... ```, unwrap it.
    If multiple fences exist, use the first.
    """
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    if m:
        return (m.group(1) or "").strip()
    return text.strip()


def extract_first_json_candidate(text: str) -> Optional[str]:
    """
    Extract the first JSON object or array substring from a messy LLM output.
    Handles:
      - leading/trailing commentary
      - code fences
      - extra text after JSON

    Returns the JSON substring (string) or None.
    """
    if not text:
        return None

    s = strip_code_fences(text)

    # Fast path: already valid JSON
    s_stripped = s.strip()
    if not s_stripped:
        return None
    if (s_stripped.startswith("{") and s_stripped.endswith("}")) or (
        s_stripped.startswith("[") and s_stripped.endswith("]")
    ):
        return s_stripped

    # Scan for first balanced {...} or [...]
    start_positions = []
    for i, ch in enumerate(s_stripped):
        if ch == "{" or ch == "[":
            start_positions.append((i, ch))
            break  # only need the first opening bracket

    if not start_positions:
        return None

    start_i, start_ch = start_positions[0]
    end_ch = "}" if start_ch == "{" else "]"

    depth = 0
    in_str = False
    escape = False

    for j in range(start_i, len(s_stripped)):
        c = s_stripped[j]

        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue

        if c == start_ch:
            depth += 1
        elif c == end_ch:
            depth -= 1
            if depth == 0:
                return s_stripped[start_i : j + 1]

    return None


def safe_json_loads_from_text(text: str) -> Optional[Any]:
    """
    Returns parsed JSON (dict/list) if we can extract & load it, else None.
    """
    cand = extract_first_json_candidate(text)
    if not cand:
        return None
    try:
        return json.loads(cand)
    except Exception:
        return None


def parse_pydantic_from_llm_text(text: str, model_cls: Type[T]) -> Optional[T]:
    """
    Extract JSON from text and validate with Pydantic model class.
    Works for both Pydantic v1 and v2.
    """
    data = safe_json_loads_from_text(text)
    if data is None:
        return None

    try:
        # pydantic v2
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            # pydantic v1
            return model_cls(**data)  # type: ignore[misc]
        except Exception:
            return None
    except Exception:
        return None
