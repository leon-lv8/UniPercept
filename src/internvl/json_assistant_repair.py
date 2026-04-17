# JSON repair for assistant schema output (shared by serve + model.chat).
from __future__ import annotations

import json
import re
from typing import Any, Match


def _strip_markdown_json_fences(blob: str) -> str:
    """Remove ``` / ```json fences and optional preamble so JSON can start at '{'."""
    s = blob.lstrip("\ufeff\u200b\u200e\u200f").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    if not s.lstrip().startswith("{"):
        m = re.search(r"```(?:json)?\s*\n?", s, flags=re.IGNORECASE)
        if m:
            s = s[m.end() :].lstrip()
    if not s.startswith("{"):
        i = s.find("{")
        if i != -1:
            s = s[i:]
    s = re.sub(r"\s*```\s*$", "", s).strip()
    return s


def _normalize_loaded_root(obj: Any) -> Any:
    """refusal / insufficient_info 必须带 aesthetic:null；模型常漏写顶层 aesthetic。"""
    if isinstance(obj, dict) and obj.get("task_type") in ("refusal", "insufficient_info"):
        if "aesthetic" not in obj:
            return {**obj, "aesthetic": None}
    return obj


def repair_assistant_json_corruption(text: str) -> str:
    if not text:
        return text
    raw = text
    s = _strip_markdown_json_fences(text)
    if not s.startswith("{"):
        return raw

    def _apply_heuristic_fixes(blob: str) -> str:
        t = blob
        t = t.replace('"weakweaknesses":', '"weaknesses":')
        t = t.replace('"weakweaknesses" :', '"weaknesses":')
        t = t.replace('"weaknesss":', '"weaknesses":')
        t = t.replace('"weaknesss" :', '"weaknesses":')
        t = t.replace('"lightlight":', '"lighting":')
        t = t.replace('"lightlight" :', '"lighting":')
        t = re.sub(r'(\n\s*\})\s*\n(\s*"subject_and_content"\s*:)', r"\1,\n\2", t)
        _ae_str_adjacent = (
            ("subject_and_content", "composition"),
            ("composition", "lighting"),
            ("lighting", "color"),
            ("color", "sharpness_and_noise"),
            ("sharpness_and_noise", "mood_and_narrative"),
            ("mood_and_narrative", "technical_issues_and_suggestions"),
            ("technical_issues_and_suggestions", "strengths"),
        )
        _root_str_adjacent = (
            ("refusal_reason", "meta"),
            ("language", "image_observed"),
            ("reasoning_brief", "limitations"),
            ("limitations", "aesthetic"),
            ("limitations", "refusal_reason"),
        )
        for a, b in _ae_str_adjacent + _root_str_adjacent:
            pat = rf'("{re.escape(a)}"\s*:\s*")([^\n]+)(\n\s*"{re.escape(b)}"\s*:)'

            def _close_prev(m: Match[str]) -> str:
                pre, mid, suf = m.group(1), m.group(2), m.group(3)
                tail = mid.rstrip()
                if tail.endswith('",'):
                    return m.group(0)
                if tail.endswith('"'):
                    return f"{pre}{mid},{suf}"
                return f'{pre}{mid}",{suf}'

            t = re.sub(pat, _close_prev, t)
        t = re.sub(
            r'("image_observed"\s*:\s*(?:true|false))(?!\s*,)(\s*\n\s*"notes"\s*:)',
            r"\1,\2",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'("aesthetic"\s*:\s*null)(?!\s*,)(\s*\n\s*"refusal_reason"\s*:)',
            r"\1,\2",
            t,
        )
        t = re.sub(r'(\])\s*\n(\s*")(weaknesses"\s*:)', r"],\n\2\3", t)
        prev = None
        while prev != t:
            prev = t
            t = re.sub(r'(:\s*")\s+"(?=[^,"\]}])', r"\1", t)
        t = t.replace('"sharp_and_noise":', '"sharpness_and_noise":')
        t = re.sub(
            r'"image_observed"\s*:\s*"true"',
            '"image_observed": true',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'"image_observed"\s*:\s*"false"',
            '"image_observed": false',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'("image_observed"\s*:\s*)true(?:\s+true)+',
            r"\1true",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'("image_observed"\s*:\s*)false(?:\s+false)+',
            r"\1false",
            t,
            flags=re.IGNORECASE,
        )
        return t

    fixed = _apply_heuristic_fixes(s)
    try:
        obj = _normalize_loaded_root(json.loads(fixed))
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        try:
            obj = _normalize_loaded_root(json.loads(s))
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            return fixed if fixed != s else raw