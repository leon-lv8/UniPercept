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
    ("mood_and_narrative", "strengths"),
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
    ('"iaaa":', '"iaa":'),
    ('"iaaa" :', '"iaa":'),
    ('"iqaa":', '"iqa":'),
    ('"iqaa" :', '"iqa":'),
    ('"iaqa":', '"iqa":'),
    ('"iaqa" :', '"iqa":'),
    ('"aaesthetic":', '"aesthetic":'),
    ('"aaesthetic" :', '"aesthetic":'),
)

# 模型偶发漏写键名闭合引号，写成 "schema_version: "1.0" 而非 "schema_version": "1.0"
_KNOWN_JSON_KEYS_MAY_MISS_CLOSING_QUOTE: Tuple[str, ...] = (
    "schema_version",
    "task_type",
    "confidence",
    "summary",
    "reasoning_brief",
    "limitations",
    "aesthetic",
    "refusal_reason",
    "meta",
    "scores",
    "subject_and_content",
    "composition",
    "lighting",
    "color",
    "sharpness_and_noise",
    "mood_and_narrative",
    "technical_issues_and_suggestions",
    "strengths",
    "weaknesses",
    "language",
    "image_observed",
    "notes",
)


def _normalize_ascii_double_quotes_for_json_repair(t: str) -> str:
    """弯引号 / 全角引号易与 JSON 不兼容，先换成 ASCII 双引号再跑纠错。"""
    return (
        t.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\uff02", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _fix_missing_closing_quote_before_colon_after_known_keys(t: str) -> str:
    """将 \"key: \" / \"key： \" 规范为 \"key\": \"（key 为白名单），避免合法 \"key\": \" 被误改。"""
    out = t
    for k in sorted(set(_KNOWN_JSON_KEYS_MAY_MISS_CLOSING_QUOTE), key=len, reverse=True):
        # 漏写键名后的闭合 \"，写成 \"schema_version: \"1.0\"；冒号可为 ASCII 或全角 \uFF1A
        pat = '"' + re.escape(k) + r"(?:\uFF1A|:)\s*\""
        out = re.sub(pat, f'"{k}": "', out)
    return out


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
    if "aaesthetic" in out and "aesthetic" not in out:
        out["aesthetic"] = out.pop("aaesthetic")
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


# 历史版本曾写入 aesthetic 的固定占位句；解析后若仍存在则删除该键，避免继续对外暴露。
_LEGACY_AESTHETIC_PLACEHOLDER_LINE = "（占位：模型未生成该项，请以 summary / reasoning_brief 为准。）"
# 历史版本曾在 meta.notes 追加的说明；对已落盘的 JSON 再经 repair 时剥除。
_LEGACY_SERVER_AESTHETIC_META_NOTE = "aesthetic 部分字段已由服务端按 schema 补全占位。"


def _strip_legacy_server_aesthetic_meta_note(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return obj
    notes = meta.get("notes")
    if not isinstance(notes, str) or _LEGACY_SERVER_AESTHETIC_META_NOTE not in notes:
        return obj
    parts = [p.strip() for p in notes.split("；") if p.strip() and p.strip() != _LEGACY_SERVER_AESTHETIC_META_NOTE]
    new_notes = "；".join(parts) if parts else "无"
    return {**obj, "meta": {**meta, "notes": new_notes}}


_AESTHETIC_STRING_KEYS: Tuple[str, ...] = (
    "subject_and_content",
    "composition",
    "lighting",
    "color",
    "sharpness_and_noise",
    "mood_and_narrative",
    "technical_issues_and_suggestions",
)


def _normalize_aesthetic_string_field(v: Any) -> str:
    """无有效文案时统一为 \"\"；剔除历史占位句。"""
    if v is None or not isinstance(v, str):
        return ""
    s = v.strip()
    if not s or s == _LEGACY_AESTHETIC_PLACEHOLDER_LINE:
        return ""
    return s


def _normalize_aesthetic_score_value(v: Any) -> Any:
    """有效数字为 float，否则为 null（键始终保留）。"""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _normalize_aesthetic_for_analysis(obj: Any) -> Any:
    """aesthetic_analysis：aesthetic 内约定键齐全；缺省 string 为 \"\"，scores 子项为 null，列表为 []。"""
    if not isinstance(obj, dict) or obj.get("task_type") != "aesthetic_analysis":
        return obj
    ae = obj.get("aesthetic")
    base_scores = {"iaa": None, "iqa": None, "ista": None}
    if ae is None or not isinstance(ae, dict):
        out_ae: dict[str, Any] = {
            "scores": dict(base_scores),
            **{k: "" for k in _AESTHETIC_STRING_KEYS},
            "strengths": [],
            "weaknesses": [],
        }
        return {**obj, "aesthetic": out_ae}
    out_ae = dict(ae)
    sc = out_ae.get("scores")
    new_scores = dict(base_scores)
    if isinstance(sc, dict):
        new_scores["iaa"] = _normalize_aesthetic_score_value(
            sc.get("iaa") if "iaa" in sc else sc.get("iaaa")
        )
        new_scores["iqa"] = _normalize_aesthetic_score_value(
            sc.get("iqa")
            if "iqa" in sc
            else (sc.get("iqaa") if "iqaa" in sc else sc.get("iaqa"))
        )
        new_scores["ista"] = _normalize_aesthetic_score_value(sc.get("ista"))
    out_ae["scores"] = new_scores
    for k in _AESTHETIC_STRING_KEYS:
        out_ae[k] = _normalize_aesthetic_string_field(out_ae.get(k))
    for k in ("strengths", "weaknesses"):
        v = out_ae.get(k)
        if not isinstance(v, list):
            out_ae[k] = []
        else:
            out_ae[k] = [x for x in v if isinstance(x, str) and x.strip()]
    return {**obj, "aesthetic": out_ae}


def _post_parse_normalize(obj: Any) -> Any:
    obj = _rename_malformed_root_keys(obj)
    obj = _normalize_loaded_root(obj)
    obj = _hoist_meta_from_nested_aesthetic(obj)
    obj = _normalize_aesthetic_for_analysis(obj)
    obj = _strip_legacy_server_aesthetic_meta_note(obj)
    obj = _ensure_meta_notes(obj)
    obj = _coerce_aesthetic_scores_numeric(obj)
    return obj


def _collapse_duplicate_commas(t: str) -> str:
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r",\s*,+", ",", t)
    return t


def _strip_spurious_comma_string_tokens(t: str) -> str:
    """
    模型偶发在字段之间插入多余的字符串 token，例如：
      "schema_version": "1.0",", "task_type": ...
    其中中间的 \",\" 是一个独立的 JSON 字符串，会导致解析失败。
    """
    prev = None
    while prev != t:
        prev = t
        # "1.0",", "task_type" -> "1.0", "task_type"（字段间多出 ,", ", ；须保留下一键的开引号）
        t = re.sub(r',\s*"\s*,\s*"\s*', ', "', t)
        # ..., ",", ...  / ..., ", ", ...
        t = re.sub(r',\s*"\s*,\s*"\s*,', ",", t)
        # ..., "," , "next_key": ...
        t = re.sub(r',\s*"\s*,\s*"\s*(?=")', ",", t)
        # ..., ", "task_type": ...  (spurious string consumes the key's opening quote)
        t = re.sub(
            r',\s*"\s*,\s*"\s*([A-Za-z_][A-Za-z0-9_]*)"\s*:',
            r', "\1":',
            t,
        )
    return t


def _quote_bare_string_array_items_for_keys(t: str, keys: Sequence[str]) -> str:
    """
    strengths/weaknesses 偶发输出未加引号的元素：
      "strengths": ["完美技术巅峰", 经典永恒主题"]
    这里会补上缺失的引号，并尽量避免误伤数字/true/false/null。
    """

    def _fix_array_body(body: str) -> str:
        b = body
        # 末项写成 , 经典永恒主题"]（缺左引号、多一个右引号），先收口再跑通用规则。
        b = re.sub(
            r',\s*([A-Za-z_\u4e00-\u9fff][^,"\]\n]*?)"\s*\]',
            lambda m: ", " + json.dumps(m.group(1).strip(), ensure_ascii=False) + "]",
            b,
        )
        b = re.sub(
            r'\[\s*([A-Za-z_\u4e00-\u9fff][^,"\]\n]*?)"\s*\]',
            lambda m: "[" + json.dumps(m.group(1).strip(), ensure_ascii=False) + "]",
            b,
        )
        # Quote first element if bare.
        b = re.sub(
            r'(\[\s*)(?!")([A-Za-z_\u4e00-\u9fff][^,\]\n"]*)(?=\s*(?:,|\]))',
            lambda m: m.group(1) + json.dumps(m.group(2).strip(), ensure_ascii=False),
            b,
        )
        # Quote subsequent elements if bare.
        b = re.sub(
            r'(,\s*)(?!")([A-Za-z_\u4e00-\u9fff][^,\]\n"]*)(?=\s*(?:,|\]))',
            lambda m: m.group(1) + json.dumps(m.group(2).strip(), ensure_ascii=False),
            b,
        )
        # If we created ..."<text>""] (extra stray quote before ]), strip the stray quote.
        b = re.sub(r'(?<=[^\s\[,])"\s*"\s*(\])', r'"\1', b)
        return b

    out = t
    for k in keys:
        # Capture array content non-greedily up to the next closing bracket.
        pat = rf'("{re.escape(k)}"\s*:\s*)(\[[\s\S]*?\])'

        def _repl(m: Match[str]) -> str:
            prefix, arr = m.group(1), m.group(2)
            # Only attempt when it still looks like an array; keep prefix unchanged.
            return prefix + _fix_array_body(arr)

        out = re.sub(pat, _repl, out)
    return out


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


def _fix_missing_comma_after_closed_string_value_before_next_key(t: str) -> str:
    """Insert missing comma when a string-valued line is immediately followed by another \"key\": line."""
    return re.sub(
        r'("[A-Za-z_][A-Za-z0-9_]*"\s*:\s*"[^"]*")\s*\r?\n(\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:)',
        r"\1,\n\2",
        t,
    )


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
    t = re.sub(
        r'("image_observed"\s*:\s*)true(?:true)+',
        r"\1true",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r'("image_observed"\s*:\s*)false(?:false)+',
        r"\1false",
        t,
        flags=re.IGNORECASE,
    )
    return t


def _heuristic_repair_text(blob: str) -> str:
    """Single pass of text-level fixes (order matters loosely; run twice for stability)."""
    t = blob
    t = _collapse_duplicate_commas(t)
    t = _strip_spurious_comma_string_tokens(t)
    t = _strip_illegal_comma_before_close(t)
    t = _normalize_ascii_double_quotes_for_json_repair(t)
    t = _fix_missing_closing_quote_before_colon_after_known_keys(t)
    t = _apply_key_alias_replacements(t)
    t = _fix_scores_block_comma_before_subject(t)
    t = _fix_missing_comma_after_closed_string_value_before_next_key(t)
    t = _fix_adjacent_string_pairs(t)
    t = _fix_structural_commas(t)
    t = _quote_bare_string_array_items_for_keys(t, ("strengths", "weaknesses"))
    t = _strip_spurious_quote_after_value_open(t)
    t = _normalize_meta_booleans(t)
    t = _collapse_duplicate_commas(t)
    t = _strip_spurious_comma_string_tokens(t)
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
    s = _normalize_ascii_double_quotes_for_json_repair(s)
    s = _fix_missing_closing_quote_before_colon_after_known_keys(s)
    s = _apply_key_alias_replacements(s)
    s = _fix_missing_comma_after_closed_string_value_before_next_key(s)
    s = _normalize_meta_booleans(s)
    s = _strip_spurious_comma_string_tokens(s)
    s = _quote_bare_string_array_items_for_keys(s, ("strengths", "weaknesses"))
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

    # fixed 可能与 s 相同（启发式对本轮输入已是稳态），但仍优于未处理的 raw：
    # 前面已对 s 做过围栏剥离、逗号/引号纠错等，绝不能因 fixed==s 而回退到 raw。
    return fixed
