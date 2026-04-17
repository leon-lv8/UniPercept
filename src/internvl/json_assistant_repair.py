# JSON repair for assistant schema output (shared by serve + model.chat).
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Match, Optional, Sequence, Tuple

_KeyPair = Tuple[str, str]

# Ordered adjacent string keys: previous line looks like "a": "<unclosed>
# and next line starts the next key; close the string before the next key.
_ROOT_STRING_ADJACENT: Sequence[_KeyPair] = (
    ("schema_version", "task_type"),
    ("task_type", "confidence"),
    ("confidence", "summary"),
    ("summary", "reasoning_brief"),
    ("reasoning_brief", "limitations"),
    ("limitations", "aesthetic"),
    ("limitations", "refusal_reason"),
    ("refusal_reason", "meta"),
)

_AESTHETIC_STRING_ADJACENT: Sequence[_KeyPair] = (
    ("subject_and_content", "composition"),
    ("composition", "lighting"),
    ("lighting", "color"),
    ("color", "sharpness_and_noise"),
    ("sharpness_and_noise", "mood_and_narrative"),
    ("mood_and_narrative", "technical_issues_and_suggestions"),
    ("technical_issues_and_suggestions", "strengths"),
)

_META_STRING_ADJACENT: Sequence[_KeyPair] = (
    ("language", "image_observed"),
    ("image_observed", "notes"),
)

# (needle, replacement) applied to raw text before JSON parse.
_KEY_ALIAS_REPLACEMENTS: Sequence[Tuple[str, str]] = (
    ('"reasonon_brief":', '"reasoning_brief":'),
    ('"reasonon_brief" :', '"reasoning_brief":'),
    ('"arefusal_reason":', '"refusal_reason":'),
    ('"arefusal_reason" :', '"refusal_reason":'),
    ('"weakweaknesses":', '"weaknesses":'),
    ('"weakweaknesses" :', '"weaknesses":'),
    ('"weaknesss":', '"weaknesses":'),
    ('"weaknesss" :', '"weaknesses":'),
    ('"lightlight":', '"lighting":'),
    ('"lightlight" :', '"lighting":'),
    ('"sharp_and_noise":', '"sharpness_and_noise":'),
    ('"sharp_and_noise" :', '"sharpness_and_noise":'),
)


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


def _hoist_meta_from_nested_aesthetic(obj: Any) -> Any:
    """模型常把 meta、refusal_reason 误写在 aesthetic 对象末尾，需挪回根对象。"""
    if not isinstance(obj, dict):
        return obj
    ae = obj.get("aesthetic")
    if not isinstance(ae, dict) or "meta" not in ae:
        return obj
    out = dict(obj)
    ae2 = dict(ae)
    meta_val = ae2.pop("meta")
    had_rr = "refusal_reason" in ae2
    rr_val = ae2.pop("refusal_reason", None) if had_rr else None
    out["aesthetic"] = ae2
    if "meta" not in out:
        out["meta"] = meta_val
    if "refusal_reason" not in out:
        out["refusal_reason"] = rr_val if had_rr else None
    return out


def _ensure_meta_notes(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    meta = obj.get("meta")
    if isinstance(meta, dict) and "notes" not in meta:
        return {**obj, "meta": {**meta, "notes": "无"}}
    return obj


def _rename_malformed_root_keys(obj: Any) -> Any:
    """解析后纠正常见错键（与字符串层 replace 双保险）。"""
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if "reasonon_brief" in out and "reasoning_brief" not in out:
        out["reasoning_brief"] = out.pop("reasonon_brief")
    if "arefusal_reason" in out and "refusal_reason" not in out:
        out["refusal_reason"] = out.pop("arefusal_reason")
    return out


def _coerce_aesthetic_scores_numeric(obj: Any) -> Any:
    """将 aesthetic.scores 中带引号的数字字符串转为 float（parse 成功后对象层修正）。"""
    if not isinstance(obj, dict) or obj.get("task_type") != "aesthetic_analysis":
        return obj
    ae = obj.get("aesthetic")
    if not isinstance(ae, dict):
        return obj
    sc = ae.get("scores")
    if not isinstance(sc, dict):
        return obj
    out_s = dict(sc)
    changed = False
    for k in ("iaa", "iqa", "ista"):
        v = out_s.get(k)
        if isinstance(v, str):
            try:
                out_s[k] = float(v.strip())
                changed = True
            except ValueError:
                pass
    if not changed:
        return obj
    return {**obj, "aesthetic": {**ae, "scores": out_s}}


def _default_aesthetic_block(obj: dict) -> dict:
    """task_type 为 aesthetic_analysis 但缺少 aesthetic 时的最小合法占位（与 schema 键齐全）。"""
    summary = (obj.get("summary") or "").strip() if isinstance(obj.get("summary"), str) else ""
    rb = (obj.get("reasoning_brief") or "").strip() if isinstance(obj.get("reasoning_brief"), str) else ""
    hint = summary[:160] if summary else (rb[:160] if rb else "模型未输出审美分项，以下为占位。")
    line = "（占位：模型未生成该项，请以 summary / reasoning_brief 为准。）"
    return {
        "scores": {"iaa": 0.0, "iqa": 0.0, "ista": 0.0},
        "subject_and_content": hint or line,
        "composition": line,
        "lighting": line,
        "color": line,
        "sharpness_and_noise": line,
        "mood_and_narrative": line,
        "technical_issues_and_suggestions": line,
        "strengths": ([hint[:100]] if len(hint) > 8 else ["待结合图像补充优点"]),
        "weaknesses": ["审美分项未完整生成"],
    }


def _ensure_aesthetic_for_analysis(obj: Any) -> Any:
    """aesthetic_analysis 必须含完整 aesthetic；补缺失键或整块缺失。"""
    if not isinstance(obj, dict) or obj.get("task_type") != "aesthetic_analysis":
        return obj
    defaults = _default_aesthetic_block(obj)
    ae = obj.get("aesthetic")
    autofilled = False
    if ae is None or not isinstance(ae, dict):
        out = dict(obj)
        out["aesthetic"] = defaults
        autofilled = True
    else:
        merged = dict(defaults)
        merged.update(ae)
        sc_def = defaults["scores"]
        sc = ae.get("scores")
        if isinstance(sc, dict):
            merged["scores"] = {**sc_def, **{k: sc[k] for k in ("iaa", "iqa", "ista") if k in sc}}
            for k in ("iaa", "iqa", "ista"):
                if k not in merged["scores"] or merged["scores"][k] is None:
                    merged["scores"][k] = 0.0
                    autofilled = True
        else:
            merged["scores"] = sc_def
            autofilled = True
        for k in (
            "subject_and_content",
            "composition",
            "lighting",
            "color",
            "sharpness_and_noise",
            "mood_and_narrative",
            "technical_issues_and_suggestions",
        ):
            v = merged.get(k)
            if not isinstance(v, str) or not v.strip():
                merged[k] = defaults[k]
                autofilled = True
        for k in ("strengths", "weaknesses"):
            v = merged.get(k)
            if not isinstance(v, list) or len(v) == 0:
                merged[k] = defaults[k]
                autofilled = True
        if merged != ae:
            autofilled = True
        out = dict(obj)
        out["aesthetic"] = merged
    if autofilled:
        meta = out.get("meta")
        if isinstance(meta, dict):
            note0 = meta.get("notes", "无")
            note0 = note0 if isinstance(note0, str) else "无"
            suffix = "aesthetic 部分字段已由服务端按 schema 补全占位。"
            if suffix not in note0:
                out = {
                    **out,
                    "meta": {**meta, "notes": f"{note0}；{suffix}" if note0.strip() else suffix},
                }
    return out


def _post_parse_normalize(obj: Any) -> Any:
    obj = _rename_malformed_root_keys(obj)
    obj = _normalize_loaded_root(obj)
    obj = _hoist_meta_from_nested_aesthetic(obj)
    obj = _ensure_aesthetic_for_analysis(obj)
    obj = _ensure_meta_notes(obj)
    obj = _coerce_aesthetic_scores_numeric(obj)
    return obj


def _collapse_duplicate_commas(t: str) -> str:
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r",\s*,+", ",", t)
    return t


def _strip_illegal_comma_before_close(t: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", t)


def _apply_key_alias_replacements(t: str) -> str:
    for needle, repl in _KEY_ALIAS_REPLACEMENTS:
        t = t.replace(needle, repl)
    return t


def _close_adjacent_unclosed_string_values(t: str, pairs: Iterable[_KeyPair]) -> str:
    """If line is '"a": "fragment' without closing quote before newline and next key b, fix it."""

    def _close_prev(m: Match[str]) -> str:
        pre, mid, suf = m.group(1), m.group(2), m.group(3)
        tail = mid.rstrip()
        if tail.endswith('",'):
            return m.group(0)
        if tail.endswith('"'):
            return f"{pre}{mid},{suf}"
        return f'{pre}{mid}",{suf}'

    out = t
    for a, b in pairs:
        pat = rf'("{re.escape(a)}"\s*:\s*")([^\n]+)(\n\s*"{re.escape(b)}"\s*:)'
        out = re.sub(pat, _close_prev, out)
    return out


def _fix_adjacent_string_pairs(t: str) -> str:
    pairs: Sequence[_KeyPair] = (
        tuple(_ROOT_STRING_ADJACENT)
        + tuple(_AESTHETIC_STRING_ADJACENT)
        + tuple(_META_STRING_ADJACENT)
    )
    return _close_adjacent_unclosed_string_values(t, pairs)


def _fix_scores_block_comma_before_subject(t: str) -> str:
    return re.sub(r'(\n\s*\})\s*\n(\s*"subject_and_content"\s*:)', r"\1,\n\2", t)


def _fix_structural_commas(t: str) -> str:
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
    return t


def _strip_spurious_quote_after_value_open(t: str) -> str:
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r'(:\s*")\s+"(?=[^,"\]}])', r"\1", t)
    return t


def _normalize_meta_booleans(t: str) -> str:
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


def _heuristic_repair_text(blob: str) -> str:
    """Single pass of text-level fixes (order matters loosely; run twice for stability)."""
    t = blob
    t = _collapse_duplicate_commas(t)
    t = _strip_illegal_comma_before_close(t)
    t = _apply_key_alias_replacements(t)
    t = _fix_scores_block_comma_before_subject(t)
    t = _fix_adjacent_string_pairs(t)
    t = _fix_structural_commas(t)
    t = _strip_spurious_quote_after_value_open(t)
    t = _normalize_meta_booleans(t)
    t = _collapse_duplicate_commas(t)
    t = _strip_illegal_comma_before_close(t)
    return t


def _try_parse_and_normalize(blob: str) -> Optional[Any]:
    try:
        return _post_parse_normalize(json.loads(blob))
    except json.JSONDecodeError:
        return None


def _trim_trailing_close_braces(blob: str, max_trim: int = 2) -> str:
    """If JSON has extra trailing '}', drop up to max_trim closing braces from the end."""
    t = blob.rstrip()
    for _ in range(max_trim + 1):
        obj = _try_parse_and_normalize(t)
        if obj is not None:
            return json.dumps(obj, ensure_ascii=False)
        if not t.endswith("}"):
            break
        t = t[:-1].rstrip()
    return blob


def repair_assistant_json_corruption(text: str) -> str:
    if not text:
        return text
    raw = text
    s = _strip_markdown_json_fences(text)
    if not s.startswith("{"):
        return raw

    obj = _try_parse_and_normalize(s)
    if obj is not None:
        return json.dumps(obj, ensure_ascii=False)

    fixed = _heuristic_repair_text(s)
    fixed = _heuristic_repair_text(fixed)

    obj = _try_parse_and_normalize(fixed)
    if obj is not None:
        return json.dumps(obj, ensure_ascii=False)

    repaired = _trim_trailing_close_braces(fixed, max_trim=2)
    if repaired != fixed:
        return repaired

    return fixed if fixed != s else raw
