from pathlib import Path

p = Path("src/internvl/json_assistant_repair.py")
s = p.read_text(encoding="utf-8")
old = r'''def _fix_missing_closing_quote_before_colon_after_known_keys(t: str) -> str:
    """将 \"key: \" 规范为 \"key\": \"（key 为白名单），避免合法 \"key\": \" 被误改。"""
    out = t
    for k in sorted(set(_KNOWN_JSON_KEYS_MAY_MISS_CLOSING_QUOTE), key=len, reverse=True):
        pat = '"' + re.escape(k) + r':\s*"'
        out = re.sub(pat, f'"{k}": "', out)
    return out'''
new = r'''def _normalize_ascii_double_quotes_for_json_repair(t: str) -> str:
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
        pat = '"' + re.escape(k) + r"(?:\uFF1A|:)\s*\""
        out = re.sub(pat, f'"{k}": "', out)
    return out'''
if old not in s:
    raise SystemExit("OLD block not found")
p.write_text(s.replace(old, new), encoding="utf-8")
print("ok")
