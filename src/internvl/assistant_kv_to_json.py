# Parse assistant line-based KV output and emit schema 1.1 JSON string.
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BEGIN_BLOCK = "BEGIN_UNIPERCEPT_KV"
END_BLOCK = "END_UNIPERCEPT_KV"

_ROOT_STRING_KEYS = (
    "schema_version",
    "task_type",
    "confidence",
    "summary",
    "reasoning_brief",
    "limitations",
)
_AESTHETIC_STRING_KEYS = (
    "subject_and_content",
    "composition",
    "lighting",
    "color",
    "sharpness_and_noise",
    "mood_and_narrative",
    "technical_issues_and_suggestions",
)
_LEGACY_AESTHETIC_PLACEHOLDER_LINE = "（占位：模型未生成该项，请以 summary / reasoning_brief 为准。）"
_LEGACY_SERVER_AESTHETIC_META_NOTE = "aesthetic 部分字段已由服务端按 schema 补全占位。"

_CN_CATEGORY_PRIMARY = frozenset(
    {"人像", "风景", "美食", "文档", "截图", "商品", "动物", "建筑", "事件", "其他"}
)
_EN_CATEGORY_PRIMARY_MAP = {
    "landscape": "风景",
    "scenery": "风景",
    "nature": "风景",
    "portrait": "人像",
    "people": "人像",
    "person": "人像",
    "human": "人像",
    "food": "美食",
    "document": "文档",
    "doc": "文档",
    "screenshot": "截图",
    "screen": "截图",
    "product": "商品",
    "commodity": "商品",
    "animal": "动物",
    "pet": "动物",
    "architecture": "建筑",
    "building": "建筑",
    "event": "事件",
    "other": "其他",
    "misc": "其他",
    "miscellaneous": "其他",
}


def _one_line_preview(s: str, max_len: int = 360) -> str:
    """用于 DEBUG 日志：折叠换行，避免多行刷屏。"""
    t = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "⏎")
    if len(t) > max_len:
        return t[: max_len - 3] + "..."
    return t


def _unescape_kv_value(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def _scrub_stray_suffix_quotes(s: str) -> str:
    """去掉句尾偶发的 \\"、孤立引号（模型或转义层噪声）。"""
    t = s.strip()
    while True:
        changed = False
        if t.endswith('\\"') or t.endswith("\\'"):
            t = t[:-2].rstrip()
            changed = True
        elif t and t[-1] in "\"'":
            t = t[:-1].rstrip()
            changed = True
        if not changed:
            break
    return t


def _sanitize_nl_value(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _scrub_stray_suffix_quotes(_unescape_kv_value(s.strip()))


def _normalize_category_primary(cp: str) -> str:
    t = _sanitize_nl_value(cp)
    if t in _CN_CATEGORY_PRIMARY:
        return t
    key = re.sub(r"[\s.,，。、]+", "", t.lower())
    return _EN_CATEGORY_PRIMARY_MAP.get(key, "其他")


def _extract_kv_body(text: str) -> str:
    t = text.lstrip("\ufeff\u200b\u200e\u200f").strip()
    if BEGIN_BLOCK in t and END_BLOCK in t:
        i0 = t.index(BEGIN_BLOCK) + len(BEGIN_BLOCK)
        i1 = t.index(END_BLOCK, i0)
        inner = t[i0:i1].strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "assistant_kv extract: used_BEGIN_END_slice inner_len=%s inner_head=%r",
                len(inner),
                _one_line_preview(inner, 200),
            )
        return inner
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "assistant_kv extract: no_BEGIN_END_pair fulltext_len=%s head=%r",
            len(t),
            _one_line_preview(t, 240),
        )
    return t


def _parse_flat_kv(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in (BEGIN_BLOCK, END_BLOCK):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.lstrip()
        if not key or not re.fullmatch(r"[a-zA-Z0-9_.]+", key):
            if key.lstrip().startswith("{"):
                logger.debug("assistant_kv: skipped non-KV line (resembles JSON): %r", raw_line[:160])
            else:
                logger.warning("assistant_kv: skipped line with invalid key: %r", raw_line[:120])
            continue
        out[key] = val
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("assistant_kv _parse_flat_kv: parsed_keys=%s", len(out))
    return out


def _parse_bool_token(s: str) -> Optional[bool]:
    x = s.strip().lower()
    if x == "true":
        return True
    if x == "false":
        return False
    return None


def _parse_null_token(s: str) -> bool:
    return s.strip().lower() == "null"


def _parse_float_or_null(s: str) -> Any:
    t = s.strip()
    if not t or _parse_null_token(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_pipe_list(s: str) -> List[str]:
    t = _unescape_kv_value(s.strip())
    if not t or t == "[]":
        return []
    return [p.strip() for p in t.split("|") if p.strip()]


def _deep_set(target: Dict[str, Any], dotted: str, leaf: Any) -> None:
    parts = dotted.split(".")
    cur: Dict[str, Any] = target
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = leaf


def _rename_malformed_root_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    if "reasonon_brief" in out and "reasoning_brief" not in out:
        out["reasoning_brief"] = out.pop("reasonon_brief")
    if "arefusal_reason" in out and "refusal_reason" not in out:
        out["refusal_reason"] = out.pop("arefusal_reason")
    if "aaesthetic" in out and "aesthetic" not in out:
        out["aesthetic"] = out.pop("aaesthetic")
    return out


def _normalize_loaded_root(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("task_type") in ("refusal", "insufficient_info"):
        if "aesthetic" not in obj:
            return {**obj, "aesthetic": None}
    return obj


def _ensure_meta_notes(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("meta")
    if isinstance(meta, dict) and "notes" not in meta:
        return {**obj, "meta": {**meta, "notes": "无"}}
    return obj


def _strip_legacy_server_aesthetic_meta_note(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("meta")
    if not isinstance(meta, dict):
        return obj
    notes = meta.get("notes")
    if not isinstance(notes, str) or _LEGACY_SERVER_AESTHETIC_META_NOTE not in notes:
        return obj
    parts = [p.strip() for p in notes.split("；") if p.strip() and p.strip() != _LEGACY_SERVER_AESTHETIC_META_NOTE]
    new_notes = "；".join(parts) if parts else "无"
    return {**obj, "meta": {**meta, "notes": new_notes}}


def _normalize_aesthetic_string_field(v: Any) -> str:
    if v is None or not isinstance(v, str):
        return ""
    s = v.strip()
    if not s or s == _LEGACY_AESTHETIC_PLACEHOLDER_LINE:
        return ""
    return s


def _normalize_aesthetic_score_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        t = v.strip()
        if not t or t.lower() == "null":
            return None
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _normalize_aesthetic_for_analysis(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("task_type") != "aesthetic_analysis":
        return obj
    ae = obj.get("aesthetic")
    base_scores = {"iaa": None, "iqa": None, "ista": None}
    if ae is None or not isinstance(ae, dict):
        out_ae: Dict[str, Any] = {
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


def _coerce_aesthetic_scores_numeric(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("task_type") != "aesthetic_analysis":
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


def _post_parse_normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj = _rename_malformed_root_keys(obj)
    obj = _normalize_loaded_root(obj)
    obj = _normalize_aesthetic_for_analysis(obj)
    obj = _strip_legacy_server_aesthetic_meta_note(obj)
    obj = _ensure_meta_notes(obj)
    obj = _coerce_aesthetic_scores_numeric(obj)
    return obj


def _parse_failure_payload(*, detail: str) -> Dict[str, Any]:
    return {
        "schema_version": "1.1",
        "task_type": "insufficient_info",
        "confidence": "low",
        "summary": "输出解析失败，已返回占位结构。",
        "reasoning_brief": "服务端无法将模型输出解析为行式 KV 契约。",
        "limitations": detail[:2000] if detail else "无",
        "aesthetic": None,
        "image_annotation": None,
        "refusal_reason": None,
        "meta": {"language": "zh-CN", "image_observed": False, "notes": "无"},
    }


def _flat_to_nested_tree(flat: Dict[str, str]) -> Dict[str, Any]:
    """Split dotted keys into nested dicts with string leaves (before typing)."""
    tree: Dict[str, Any] = {}
    for k, v in flat.items():
        if k in ("aesthetic", "image_annotation", "refusal_reason"):
            continue
        _deep_set(tree, k, v)
    return tree


def _coerce_meta(meta_raw: Any) -> Dict[str, Any]:
    if not isinstance(meta_raw, dict):
        return {"language": "zh-CN", "image_observed": False, "notes": "无"}
    lang = meta_raw.get("language", "zh-CN")
    lang_s = lang if isinstance(lang, str) and lang.strip() else "zh-CN"
    obs = meta_raw.get("image_observed", False)
    if isinstance(obs, str):
        b = _parse_bool_token(obs)
        obs_b = bool(b) if b is not None else False
    else:
        obs_b = bool(obs) if isinstance(obs, bool) else False
    notes = meta_raw.get("notes", "无")
    notes_s = _sanitize_nl_value(str(notes)) if isinstance(notes, str) else "无"
    if not notes_s.strip():
        notes_s = "无"
    return {"language": lang_s, "image_observed": obs_b, "notes": notes_s}


def _coerce_safety_flags(sf: Any) -> Dict[str, bool]:
    base = {"adult": False, "violence": False, "medical": False, "sensitive": False}
    if not isinstance(sf, dict):
        return dict(base)
    out = dict(base)
    for k in out:
        v = sf.get(k, False)
        if isinstance(v, str):
            b = _parse_bool_token(v)
            out[k] = bool(b) if b is not None else False
        else:
            out[k] = bool(v) if isinstance(v, bool) else False
    return out


def _build_image_annotation_from_tree(sub: Any) -> Optional[Dict[str, Any]]:
    if sub is None:
        return None
    if isinstance(sub, str) and _parse_null_token(sub):
        return None
    if not isinstance(sub, dict):
        return None
    tags_raw = sub.get("tags", "")
    if isinstance(tags_raw, list):
        tags = [_sanitize_nl_value(str(x)) for x in tags_raw if isinstance(x, str) and x.strip()]
    else:
        tags = [_sanitize_nl_value(x) for x in _parse_pipe_list(str(tags_raw))]
    cp = _normalize_category_primary(str(sub.get("category_primary", "")))
    cs_raw = sub.get("category_secondary", "null")
    if isinstance(cs_raw, str):
        tcs = cs_raw.strip()
        if not tcs or tcs.lower() == "null":
            cs = None
        else:
            cs = _sanitize_nl_value(tcs)
    else:
        cs = None
    desc_s = _sanitize_nl_value(str(sub.get("description_short", "")))
    desc_d = _sanitize_nl_value(str(sub.get("description_detailed", "")))
    ocr_v = sub.get("ocr_text", "null")
    if isinstance(ocr_v, str) and (not ocr_v.strip() or ocr_v.strip().lower() == "null"):
        ocr = None
    elif ocr_v is None:
        ocr = None
    else:
        ocr = _sanitize_nl_value(str(ocr_v)) or None
    sf_sub = sub.get("safety_flags")
    if isinstance(sf_sub, dict):
        sf = _coerce_safety_flags(sf_sub)
    else:
        sf = _coerce_safety_flags({})
    return {
        "category_primary": cp,
        "category_secondary": cs,
        "tags": tags,
        "description_short": desc_s,
        "description_detailed": desc_d,
        "ocr_text": ocr,
        "safety_flags": sf,
    }


def _build_aesthetic_from_tree(sub: Any) -> Any:
    if sub is None:
        return None
    if isinstance(sub, str) and _parse_null_token(sub):
        return None
    if not isinstance(sub, dict):
        return None
    scores_raw = sub.get("scores", {})
    scores: Dict[str, Any] = {"iaa": None, "iqa": None, "ista": None}
    if isinstance(scores_raw, dict):
        for m in ("iaa", "iqa", "ista"):
            scores[m] = _parse_float_or_null(str(scores_raw.get(m, "null")))
    ae: Dict[str, Any] = {"scores": scores}
    for k in _AESTHETIC_STRING_KEYS:
        v = sub.get(k, "")
        ae[k] = _sanitize_nl_value(str(v)) if v is not None else ""
    st = sub.get("strengths", "")
    wk = sub.get("weaknesses", "")
    if isinstance(st, list):
        ae["strengths"] = [_sanitize_nl_value(str(x)) for x in st if isinstance(x, str) and x.strip()]
    else:
        ae["strengths"] = [_sanitize_nl_value(x) for x in _parse_pipe_list(str(st))]
    if isinstance(wk, list):
        ae["weaknesses"] = [_sanitize_nl_value(str(x)) for x in wk if isinstance(x, str) and x.strip()]
    else:
        ae["weaknesses"] = [_sanitize_nl_value(x) for x in _parse_pipe_list(str(wk))]
    return ae


def flat_kv_to_root_object(flat: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (obj, error_detail). obj is None on hard failure.
    """
    if not flat:
        return None, "empty_kv"

    tree = _flat_to_nested_tree(flat)

    # Whole-object null lines override nested keys
    if "aesthetic" in flat and _parse_null_token(flat["aesthetic"]):
        aesthetic_val: Any = None
    else:
        aesthetic_val = _build_aesthetic_from_tree(tree.get("aesthetic"))

    if "image_annotation" in flat and _parse_null_token(flat["image_annotation"]):
        image_ann: Any = None
    else:
        image_ann = _build_image_annotation_from_tree(tree.get("image_annotation"))

    root: Dict[str, Any] = {}
    for k in _ROOT_STRING_KEYS:
        raw = flat.get(k, "")
        raw_s = raw.strip() if raw is not None else ""
        if k in ("schema_version", "task_type", "confidence"):
            root[k] = _unescape_kv_value(raw_s)
        elif k == "limitations" and _parse_null_token(raw_s):
            root[k] = None
        else:
            root[k] = _sanitize_nl_value(raw_s)
    if not str(root.get("schema_version", "")).strip():
        root["schema_version"] = "1.1"

    rr_raw = flat.get("refusal_reason", "null")
    if _parse_null_token(str(rr_raw)):
        root["refusal_reason"] = None
    else:
        rr_s = _sanitize_nl_value(str(rr_raw))
        root["refusal_reason"] = rr_s or None

    root["aesthetic"] = aesthetic_val
    root["image_annotation"] = image_ann
    root["meta"] = _coerce_meta(tree.get("meta"))

    task = str(root.get("task_type", "")).strip()
    if task not in ("general_qa", "aesthetic_analysis", "refusal", "insufficient_info"):
        return None, f"invalid_task_type:{task!r}"

    conf = str(root.get("confidence", "")).strip()
    if conf not in ("high", "medium", "low"):
        return None, f"invalid_confidence:{conf!r}"

    if task != "aesthetic_analysis":
        root["aesthetic"] = None
    elif root["aesthetic"] is None:
        root["aesthetic"] = _build_aesthetic_from_tree({})

    if task == "refusal":
        if not isinstance(root.get("refusal_reason"), str) or not str(root["refusal_reason"]).strip():
            root["refusal_reason"] = "拒答"

    if task != "refusal" and root.get("refusal_reason") is not None:
        root["refusal_reason"] = None

    root = _post_parse_normalize(root)

    # Final required root keys guard
    for k in (
        "schema_version",
        "task_type",
        "confidence",
        "summary",
        "reasoning_brief",
        "limitations",
        "aesthetic",
        "image_annotation",
        "refusal_reason",
        "meta",
    ):
        if k not in root:
            return None, f"missing_root_key:{k}"

    return root, None


def assistant_kv_text_to_obj(text: str) -> Dict[str, Any]:
    raw = text.lstrip("\ufeff\u200b\u200e\u200f")
    stripped = raw.strip()
    lines = stripped.splitlines()
    first_nl = lines[0] if lines else ""
    last_nl = lines[-1] if lines else ""
    has_markers = BEGIN_BLOCK in stripped and END_BLOCK in stripped
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "assistant_kv enter: raw_len=%s line_count=%s has_BEGIN_END=%s "
            "first_starts_BEGIN=%s first_starts_brace=%s first_line=%r last_line=%r",
            len(stripped),
            len(lines),
            has_markers,
            first_nl.strip().startswith(BEGIN_BLOCK),
            stripped.startswith("{"),
            first_nl[:200],
            last_nl[:120],
        )
        logger.debug("assistant_kv enter preview: %s", _one_line_preview(stripped, 400))

    body = _extract_kv_body(text)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "assistant_kv body: len=%s line_count=%s preview=%s",
            len(body),
            len(body.splitlines()),
            _one_line_preview(body, 400),
        )

    flat = _parse_flat_kv(body)
    if logger.isEnabledFor(logging.DEBUG):
        ks = sorted(flat.keys())
        logger.debug(
            "assistant_kv flat: key_count=%s keys_sample=%s",
            len(flat),
            ks[:30] if len(ks) > 30 else ks,
        )

    obj, err = flat_kv_to_root_object(flat)
    if obj is None:
        detail = err or "parse_error"
        if detail == "empty_kv" and body.lstrip().startswith("{"):
            logger.warning(
                "assistant_kv: empty_kv 且正文以 `{` 开头，疑似模型仍输出 JSON。"
                "请确认容器内 SYSTEM_PROMPT_FILE 指向 config/system_prompt_kv.txt 并已重载提示词。"
            )
        else:
            logger.warning("assistant_kv parse failed: %s", detail)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "assistant_kv fail detail=%r body_head=%r",
                detail,
                _one_line_preview(body, 500),
            )
        return _parse_failure_payload(detail=f"KV解析失败：{detail}")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "assistant_kv ok: task_type=%s confidence=%s image_observed=%s",
            obj.get("task_type"),
            obj.get("confidence"),
            (obj.get("meta") or {}).get("image_observed"),
        )
    return obj


def assistant_kv_text_to_json_str(text: str) -> str:
    return json.dumps(assistant_kv_text_to_obj(text), ensure_ascii=False)
