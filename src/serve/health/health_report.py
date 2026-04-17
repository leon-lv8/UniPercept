from __future__ import annotations

import asyncio
import csv
import html
import io
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..runtime.model_load import _inference_profile_snapshot
from ..runtime.state import STATE


def _described(value: Any, description_zh: str) -> Dict[str, str | Any]:
    return {"value": value, "description": description_zh}


_HEALTH_TOP_DESC: Dict[str, str] = {
    "status": "服务健康状态；ok 表示 HTTP 服务可用。",
    "model_loaded": "模型权重是否已完成加载并可处理推理请求。",
    "model_loading": "模型是否仍在后台加载（为 true 时推理接口会返回 503）。",
    "model_load_error": "模型加载失败时的错误摘要；成功或未失败时为 null。",
    "device": "推理使用的 PyTorch 设备（如 cuda:0 或 cpu）。",
    "model_id": "对外暴露的模型标识（可与 OpenAI 兼容客户端中的 model 字段对应）。",
    "load_seconds": "启动阶段加载模型所耗时间（秒）。",
    "gpu_memory_used_mib": "服务侧查询到的当前 GPU 显存已用量（MiB，用于快速验证省显存效果）。",
    "gpu_power_draw_w": "服务侧 nvidia-smi 查询到的当前 GPU 功耗读数（瓦）；空闲时可能为 [N/A]（取首块 GPU）。",
    "gpu_power_limit_w": "服务侧 nvidia-smi 查询到的 GPU 功耗上限（瓦）（取首块 GPU）。",
    "inference_profile": "推理与视觉管线相关的关键配置快照（便于排查性能与显存问题）。",
    "gpu": "通过 nvidia-smi 查询到的 NVIDIA GPU 状态（无 GPU 或命令不可用时见子字段说明）。",
}

_INFERENCE_PROFILE_DESC: Dict[str, str] = {
    "param_dtype": "模型参数的数据类型（例如 bfloat16、float32）。",
    "load_in_8bit": "是否启用 8bit 量化权重加载（bitsandbytes）。",
    "load_in_4bit": "是否启用 4bit 量化权重加载（bitsandbytes）。",
    "llm_attn_implementation": "语言模型注意力实现方式（如 flash_attention_2、eager 等）。",
    "vision_use_flash_attn": "视觉编码器（ViT）侧是否启用 FlashAttention 类加速实现。",
    "config_effective_image_size": "配置中生效的输入图像边长（像素），来自 force_image_size 或 vision_config。",
    "vision_input_size_effective": "图像预处理实际使用的边长；可被环境变量 VISION_INPUT_SIZE 覆盖。",
    "inference_use_cache": "是否在推理阶段启用 KV cache（INFERENCE_USE_CACHE）。",
    "max_new_tokens": "单次对话生成时允许的新增 token 数量上限。",
    "max_images_per_request": "单个请求中允许附带的最大图像数量。",
    "max_prompt_total_chars": "环境变量 MAX_PROMPT_TOTAL_CHARS 的原始值；0 表示不按字符数截断提示。",
}

_NVIDIA_SMI_QUERY = (
    "index,name,memory.total,memory.used,memory.free,temperature.gpu,"
    "utilization.gpu,utilization.memory,uuid,pci.bus_id,power.draw,power.limit,driver_version"
)

# 顺序必须与 _NVIDIA_SMI_QUERY 的 CSV 列顺序一致（供 _nvidia_smi_gpu_devices 解析）。
_NVIDIA_SMI_FIELD_KEYS = [
    "index",
    "name",
    "memory_total_mib",
    "memory_used_mib",
    "memory_free_mib",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "uuid",
    "pci_bus_id",
    "power_draw_w",
    "power_limit_w",
    "driver_version",
]

# health 表格与 devices JSON 内字段的展示顺序（解析仍用上方列表）。
_NVIDIA_SMI_FIELD_KEYS_DISPLAY = [
    "memory_total_mib",
    "memory_used_mib",
    "memory_free_mib",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "power_draw_w",
    "power_limit_w",
    "index",
    "name",
    "uuid",
    "pci_bus_id",
    "driver_version",
]

_GPU_SECTION_DESC: Dict[str, str] = {
    "nvidia_smi_ok": "是否成功执行 nvidia-smi 并解析到至少一块 GPU。",
    "nvidia_smi_error": "当查询失败时，简要错误信息；成功时为 null。",
    "devices": "各 GPU 的静态与实时指标列表（数值来自 nvidia-smi 查询时刻）。",
}

# /health HTML 页面：字段短标题（中文），说明文字复用上方 *_DESC 字典。
_HEALTH_HTML_LABELS: Dict[str, str] = {
    "status": "服务状态",
    "model_loaded": "模型已加载",
    "model_loading": "模型加载中",
    "model_load_error": "模型加载错误",
    "device": "推理设备",
    "gpu_memory_used_mib": "GPU 显存已用 (MiB)",
    "gpu_power_draw_w": "GPU 当前功耗 (W)",
    "gpu_power_limit_w": "GPU 功耗上限 (W)",
    "model_id": "模型标识",
    "load_seconds": "模型加载耗时",
    "inference_profile": "推理与视觉配置",
    "gpu": "GPU 状态",
}

_GPU_SUB_HTML_LABELS: Dict[str, str] = {
    "nvidia_smi_ok": "nvidia-smi 是否成功",
    "nvidia_smi_error": "nvidia-smi 错误说明",
    "devices": "GPU 设备明细",
}

_NVIDIA_SMI_FIELD_LABELS_ZH: Dict[str, str] = {
    "index": "设备索引",
    "name": "型号名称",
    "memory_total_mib": "显存总量 (MiB)",
    "memory_used_mib": "显存已用 (MiB)",
    "memory_free_mib": "显存空闲 (MiB)",
    "temperature_gpu_c": "核心温度 (°C)",
    "utilization_gpu_pct": "GPU 利用率 (%)",
    "utilization_memory_pct": "显存利用率 (%)",
    "uuid": "UUID",
    "pci_bus_id": "PCI 总线 ID",
    "power_draw_w": "当前功耗 (W)",
    "power_limit_w": "功耗上限 (W)",
    "driver_version": "驱动版本",
}

_INFERENCE_PROFILE_LABELS_ZH: Dict[str, str] = {
    "param_dtype": "参数精度",
    "load_in_8bit": "8bit 量化加载",
    "load_in_4bit": "4bit 量化加载",
    "llm_attn_implementation": "LLM 注意力实现",
    "vision_use_flash_attn": "ViT FlashAttention",
    "config_effective_image_size": "配置图像边长 (px)",
    "vision_input_size_effective": "预处理图像边长 (px)",
    "inference_use_cache": "推理 KV Cache",
    "max_new_tokens": "最大新生成 token",
    "max_images_per_request": "单请求最大图片数",
    "max_prompt_total_chars": "提示最大字符数配置",
}

_NVIDIA_SMI_LEAF_DESC: Dict[str, str] = {
    "index": "GPU 设备索引。",
    "name": "GPU 产品名称/型号。",
    "memory_total_mib": "显存总容量（MiB）。",
    "memory_used_mib": "当前已使用显存（MiB）。",
    "memory_free_mib": "当前空闲显存（MiB）。",
    "temperature_gpu_c": "GPU 核心温度（摄氏度）。",
    "utilization_gpu_pct": "GPU 计算利用率（%）。",
    "utilization_memory_pct": "显存控制器利用率（%）。",
    "uuid": "GPU 唯一标识符（UUID）。",
    "pci_bus_id": "PCI 总线 ID。",
    "power_draw_w": "当前功耗读数（瓦）；部分空闲状态下可能为 [N/A]。",
    "power_limit_w": "功耗上限（瓦）。",
    "driver_version": "NVIDIA 驱动版本号（与具体 GPU 行重复属 nvidia-smi 正常行为）。",
}


def _coerce_gpu_csv_field(key: str, raw: str) -> Any:
    t = raw.strip()
    if t in {"", "[N/A]", "N/A", "[Unknown Error]"}:
        return None
    if key in ("name", "uuid", "pci_bus_id", "driver_version"):
        return t
    if key == "index":
        try:
            return int(t)
        except ValueError:
            return t
    try:
        return float(t) if "." in t else int(t)
    except ValueError:
        return t


def _nvidia_smi_gpu_devices() -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """Run nvidia-smi once; return (ok, error_message_or_none, list of flat gpu dicts)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_SMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=float(os.environ.get("NVIDIA_SMI_TIMEOUT_SEC", "8")),
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi 未找到（可能未安装 NVIDIA 驱动或未在 PATH 中）", []
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi 执行超时", []

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        return False, err[:500], []

    lines = [ln.strip() for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    if not lines:
        return False, "nvidia-smi 无输出（可能无可用 GPU）", []

    devices: List[Dict[str, Any]] = []
    for line in lines:
        row = next(csv.reader(io.StringIO(line)))
        if len(row) < len(_NVIDIA_SMI_FIELD_KEYS):
            continue
        d: Dict[str, Any] = {}
        for i, key in enumerate(_NVIDIA_SMI_FIELD_KEYS):
            d[key] = _coerce_gpu_csv_field(key, row[i])
        devices.append(d)

    if not devices:
        return False, "未能解析 nvidia-smi 的 CSV 输出", []
    return True, None, devices


def _wrap_inference_profile(prof: Dict[str, Any]) -> Dict[str, Dict[str, str | Any]]:
    return {k: _described(v, _INFERENCE_PROFILE_DESC.get(k, f"配置项「{k}」。")) for k, v in prof.items()}


def _wrap_gpu_devices(devices: List[Dict[str, Any]]) -> List[Dict[str, Dict[str, str | Any]]]:
    out: List[Dict[str, Dict[str, str | Any]]] = []
    for dev in devices:
        out.append(
            {k: _described(dev[k], _NVIDIA_SMI_LEAF_DESC[k]) for k in _NVIDIA_SMI_FIELD_KEYS_DISPLAY if k in dev}
        )
    return out


_HEALTH_AUTO_REFRESH_SCRIPT = """<script>
(function () {
  var K_ON = "unipercept_health_autorefresh";
  var K_SEC = "unipercept_health_refresh_interval_sec";
  var DEF = 30, MIN = 3, MAX = 3600;
  function clamp(n) {
    n = parseInt(String(n), 10);
    if (isNaN(n)) return DEF;
    return Math.min(MAX, Math.max(MIN, n));
  }
  var cb = document.getElementById("arOn");
  var inp = document.getElementById("arSec");
  var meta = document.getElementById("arMeta");
  var timer = null, cd = 0;
  function loadOn() {
    try { var v = localStorage.getItem(K_ON); if (v === null) return false; return v === "1" || v === "true"; }
    catch (e) { return false; }
  }
  function saveOn(on) { try { localStorage.setItem(K_ON, on ? "1" : "0"); } catch (e) {} }
  function loadSec() {
    try { var v = localStorage.getItem(K_SEC); if (v === null) return DEF; return clamp(v); }
    catch (e) { return DEF; }
  }
  function saveSec() { try { localStorage.setItem(K_SEC, String(clamp(inp.value))); } catch (e) {} }
  function clearT() { if (timer !== null) { clearInterval(timer); timer = null; } }
  function metaTxt() {
    if (!meta) return;
    if (!cb || !cb.checked) { meta.textContent = ""; return; }
    meta.textContent = "约 " + cd + " 秒后刷新";
  }
  function arm() {
    clearT();
    if (!cb || !inp || !cb.checked) { metaTxt(); return; }
    var sec = clamp(inp.value);
    inp.value = String(sec);
    saveSec();
    cd = sec;
    metaTxt();
    timer = setInterval(function () {
      cd -= 1;
      if (cd <= 0) { clearT(); location.reload(); return; }
      metaTxt();
    }, 1000);
  }
  if (cb && inp) {
    var qs = new URLSearchParams(window.location.search || "");
    var qOn = qs.get("ar");
    var qSec = qs.get("ar_sec");
    if (qOn === "1" || qOn === "true") { cb.checked = true; }
    if (qOn === "0" || qOn === "false") { cb.checked = false; }
    if (qOn === null || qOn === "") { cb.checked = loadOn(); }
    else { saveOn(cb.checked); }
    if (qSec !== null && qSec !== "") { inp.value = String(clamp(qSec)); saveSec(); }
    else { inp.value = String(loadSec()); }
    inp.addEventListener("change", arm);
    inp.addEventListener("input", arm);
    cb.addEventListener("change", function () { saveOn(cb.checked); arm(); });
    arm();
  }
})();
</script>"""


def _unwrap_described(obj: Any) -> Any:
    """Strip {value, description} wrappers used by /health JSON for machine clients."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"value", "description"}:
            return _unwrap_described(obj["value"])
        return {k: _unwrap_described(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_described(i) for i in obj]
    return obj


def _health_wants_json(request: Request) -> bool:
    if request.query_params.get("format") == "json":
        return True
    accept = (request.headers.get("accept") or "").lower()
    return "application/json" in accept


def _health_is_described(node: Any) -> bool:
    return isinstance(node, dict) and set(node.keys()) == {"value", "description"}


def _health_desc_value(node: Any) -> Any:
    if _health_is_described(node):
        return node["value"]
    return node


def _health_desc_help(node: Any) -> str:
    if _health_is_described(node):
        return str(node.get("description") or "")
    return ""


def _health_html_page(
    out: Dict[str, Any],
    *,
    model_path: str = "",
    max_new_tokens_effective: Any = None,
    page_generated_at: str = "",
    cuda_available: Optional[bool] = None,
    cuda_device_name: Optional[str] = None,
) -> str:
    """浏览器健康页：中文名称与说明，含推理配置与完整 GPU 列。"""
    plain = _unwrap_described(out)
    safe_json = html.escape(json.dumps(plain, ensure_ascii=False, indent=2), quote=False)

    def cell(x: Any) -> str:
        if x is None:
            return "—"
        return html.escape(str(x), quote=False)

    def row_zh(label_zh: str, desc_zh: str, val: Any) -> str:
        lh = html.escape(label_zh, quote=False)
        dh = html.escape(desc_zh, quote=False) if desc_zh else ""
        return (
            f'<tr><th scope="row"><span class="lbl">{lh}</span>'
            f'<div class="desc">{dh}</div></th>'
            f'<td class="val">{cell(val)}</td></tr>'
        )

    main_keys = [
        "status",
        "model_loaded",
        "model_loading",
        "model_load_error",
        "device",
        "gpu_memory_used_mib",
        "gpu_power_draw_w",
        "gpu_power_limit_w",
        "model_id",
        "load_seconds",
    ]
    tr_main = ""
    for key in main_keys:
        if key not in out:
            continue
        node = out[key]
        lbl = _HEALTH_HTML_LABELS.get(key, key)
        hint = _health_desc_help(node) or _HEALTH_TOP_DESC.get(key, "")
        tr_main += row_zh(lbl, hint, _health_desc_value(node))

    if model_path:
        tr_main += row_zh(
            "模型权重路径",
            "来自环境变量 MODEL_PATH，指向本服务加载的权重目录。",
            model_path,
        )
    if max_new_tokens_effective is not None:
        tr_main += row_zh(
            "默认单次最大生成长度",
            "对应生成配置中的 max_new_tokens；客户端仍可在请求里用 max_tokens 覆盖。",
            max_new_tokens_effective,
        )
    if page_generated_at:
        tr_main += row_zh(
            "本页生成时间",
            "服务端本地时间，用于判断信息新鲜度。",
            page_generated_at,
        )
    if cuda_available is not None:
        tr_main += row_zh(
            "PyTorch 可见 CUDA",
            "torch.cuda.is_available() 的查询结果。",
            "是" if cuda_available else "否",
        )
    if cuda_device_name:
        tr_main += row_zh(
            "CUDA 设备名称",
            "torch.cuda.get_device_name 返回的当前索引对应 GPU 名称。",
            cuda_device_name,
        )
    tr_main += row_zh(
        "PyTorch 版本",
        "当前进程内已加载的 torch 软件包版本。",
        getattr(torch, "__version__", "—"),
    )
    tr_main += row_zh(
        "Python 版本",
        "运行本服务的解释器主、次、修订版本号。",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    tr_gpu_meta = ""
    devices: List[Any] = []  # nvidia-smi device rows
    gpu_block = out.get("gpu")
    gv = _health_desc_value(gpu_block) if gpu_block is not None else None
    if isinstance(gv, dict):
        for subk in ("nvidia_smi_ok", "nvidia_smi_error"):
            if subk not in gv:
                continue
            sn = gv[subk]
            lbl = _GPU_SUB_HTML_LABELS.get(subk, subk)
            hint = _health_desc_help(sn) or _GPU_SECTION_DESC.get(subk, "")
            tr_gpu_meta += row_zh(lbl, hint, _health_desc_value(sn))
        dev_node = gv.get("devices")
        raw_dev = _health_desc_value(dev_node)
        if isinstance(raw_dev, list):
            devices = raw_dev

    hdr_cells = []
    for k in _NVIDIA_SMI_FIELD_KEYS_DISPLAY:
        lbl = _NVIDIA_SMI_FIELD_LABELS_ZH.get(k, k)
        dsc = _NVIDIA_SMI_LEAF_DESC.get(k, "")
        hdr_cells.append(
            f'<th><span class="lbl">{html.escape(lbl, quote=False)}</span>'
            f'<div class="desc">{html.escape(dsc, quote=False)}</div></th>'
        )
    hdr_gpu = "".join(hdr_cells)
    body_gpu = ""
    ncol = len(_NVIDIA_SMI_FIELD_KEYS_DISPLAY)
    if devices:
        for d in devices:
            if not isinstance(d, dict):
                body_gpu += f'<tr><td class="val" colspan="{ncol}">{cell(d)}</td></tr>'
                continue
            tds = []
            for k in _NVIDIA_SMI_FIELD_KEYS_DISPLAY:
                tds.append(f'<td class="val">{cell(_health_desc_value(d.get(k)))}</td>')
            body_gpu += "<tr>" + "".join(tds) + "</tr>"
    else:
        body_gpu = f'<tr><td class="val" colspan="{ncol}">无可用设备或 nvidia-smi 查询失败</td></tr>'

    tr_prof = ""
    prof_block = out.get("inference_profile")
    pv = _health_desc_value(prof_block)
    if isinstance(pv, dict) and pv:
        for key in sorted(pv.keys()):
            node = pv[key]
            lbl = _INFERENCE_PROFILE_LABELS_ZH.get(key, key)
            hint = _health_desc_help(node) or _INFERENCE_PROFILE_DESC.get(key, "")
            tr_prof += row_zh(lbl, hint, _health_desc_value(node))
    else:
        tr_prof = (
            '<tr><th scope="row"><span class="lbl">说明</span>'
            '<div class="desc">模型未就绪时尚无快照。</div></th>'
            '<td class="val">—</td></tr>'
        )

    # f-string 的 {{}} 表达式内不能含反斜杠转义；fallback 单独定义。
    _gpu_meta_empty_row = '<tr><td class="val" colspan="2">—</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>UniPercept · 健康检查</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1419; color: #e6edf3; }}
    h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 1rem; }}
    h2 {{ font-size: 1rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: #c9d1d9; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 56rem; margin-bottom: 1rem; }}
    th, td {{ text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid #30363d; vertical-align: top; }}
    th {{ width: 13rem; color: #8b949e; font-weight: 500; }}
    .lbl {{ display: block; color: #c9d1d9; }}
    .desc {{ font-size: 0.72rem; color: #7d8590; margin-top: 0.25rem; line-height: 1.35; max-width: 22rem; }}
    .val {{ color: #e6edf3; }}
    caption {{ text-align: left; font-size: 0.85rem; color: #8b949e; margin-bottom: 0.35rem; }}
    .scroll-x {{ overflow-x: auto; max-width: 100%; margin-bottom: 1rem; }}
    .tbl-wide th {{ min-width: 6.5rem; width: auto; }}
    pre {{ background: #161b22; padding: 1rem; border-radius: 6px; overflow: auto; max-width: 56rem;
           font-size: 0.78rem; line-height: 1.45; border: 1px solid #30363d; }}
    .hint {{ font-size: 0.8rem; color: #8b949e; margin-top: 1rem; max-width: 56rem; line-height: 1.45; }}
    code {{ font-size: 0.85em; background: #21262d; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .refresh-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; margin: 0 0 1.25rem;
      padding: 0.65rem 0.85rem; background: #161b22; border: 1px solid #30363d; border-radius: 6px; max-width: 56rem; }}
    .refresh-bar label {{ display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.88rem; color: #c9d1d9; cursor: pointer; }}
    .refresh-bar input[type="number"] {{
      width: 5rem; padding: 0.25rem 0.4rem; border-radius: 4px; border: 1px solid #30363d; background: #0d1117; color: #e6edf3; }}
    .refresh-meta {{ font-size: 0.78rem; color: #8b949e; }}
  </style>
</head>
<body>
  <h1>UniPercept 服务健康检查</h1>
  <div class="refresh-bar" id="refreshBar">
    <label><input type="checkbox" id="arOn"/> 自动刷新页面</label>
    <label>刷新间隔（秒）<input type="number" id="arSec" min="3" max="3600" step="1" value="30"/></label>
    <span class="refresh-meta" id="arMeta"></span>
  </div>
  <table>
    <caption>服务与模型概要</caption>
    <tbody>{tr_main}</tbody>
  </table>
  <h2>GPU 查询结果</h2>
  <table>
    <caption>nvidia-smi 概要</caption>
    <tbody>{tr_gpu_meta or _gpu_meta_empty_row}</tbody>
  </table>
  <div class="scroll-x">
    <table class="tbl-wide">
      <caption>GPU 设备明细（各列含义见表头下方说明）</caption>
      <thead><tr>{hdr_gpu}</tr></thead>
      <tbody>{body_gpu}</tbody>
    </table>
  </div>
  <h2>推理与视觉配置</h2>
  <table>
    <caption>与当前加载模型相关的关键参数</caption>
    <tbody>{tr_prof}</tbody>
  </table>
  <p class="hint">自动刷新开关与间隔（秒）保存在浏览器 <code>localStorage</code>；也可用 <code>?ar=1&amp;ar_sec=15</code> 在打开页面时临时指定是否开启与间隔。下方为与 JSON 接口一致的<strong>完整原始数据</strong>（字段名为英文，便于脚本解析）。若只要 JSON 响应，请使用请求头 <code>Accept: application/json</code> 或查询参数 <code>?format=json</code>。</p>
  <pre>{safe_json}</pre>
{_HEALTH_AUTO_REFRESH_SCRIPT}
</body>
</html>"""


async def health(request: Request):
    ok_smi, smi_err, gpu_devices = await asyncio.to_thread(_nvidia_smi_gpu_devices)
    prof = _inference_profile_snapshot()
    mem_used_mib = None
    power_draw_w = None
    power_limit_w = None
    if ok_smi and gpu_devices:
        g0 = gpu_devices[0]
        mem_used_mib = g0.get("memory_used_mib")
        power_draw_w = g0.get("power_draw_w")
        power_limit_w = g0.get("power_limit_w")

    out: Dict[str, Any] = {
        "status": _described("ok", _HEALTH_TOP_DESC["status"]),
        "model_loaded": _described(STATE.model is not None, _HEALTH_TOP_DESC["model_loaded"]),
        "model_loading": _described(STATE.model_loading, _HEALTH_TOP_DESC["model_loading"]),
        "model_load_error": _described(STATE.model_load_error, _HEALTH_TOP_DESC["model_load_error"]),
        "device": _described(str(STATE.device), _HEALTH_TOP_DESC["device"]),
        "gpu_memory_used_mib": _described(mem_used_mib, _HEALTH_TOP_DESC["gpu_memory_used_mib"]),
        "gpu_power_draw_w": _described(power_draw_w, _HEALTH_TOP_DESC["gpu_power_draw_w"]),
        "gpu_power_limit_w": _described(power_limit_w, _HEALTH_TOP_DESC["gpu_power_limit_w"]),
        "model_id": _described(STATE.model_id, _HEALTH_TOP_DESC["model_id"]),
        "load_seconds": _described(getattr(request.app.state, "load_seconds", None), _HEALTH_TOP_DESC["load_seconds"]),
        "gpu": _described(
            {
                "nvidia_smi_ok": _described(ok_smi, _GPU_SECTION_DESC["nvidia_smi_ok"]),
                "nvidia_smi_error": _described(smi_err, _GPU_SECTION_DESC["nvidia_smi_error"]),
                "devices": _described(_wrap_gpu_devices(gpu_devices), _GPU_SECTION_DESC["devices"]),
            },
            _HEALTH_TOP_DESC["gpu"],
        ),
    }
    if prof:
        out["inference_profile"] = _described(_wrap_inference_profile(prof), _HEALTH_TOP_DESC["inference_profile"])
    if _health_wants_json(request):
        return JSONResponse(out)
    max_nt = STATE.gen_cfg.get("max_new_tokens") if STATE.gen_cfg else None
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cuda_ok = bool(torch.cuda.is_available())
    cuda_nm: Optional[str] = None
    if cuda_ok and STATE.device is not None and STATE.device.type == "cuda":
        try:
            di = STATE.device.index if STATE.device.index is not None else 0
            cuda_nm = str(torch.cuda.get_device_name(di))
        except Exception:
            cuda_nm = None
    return HTMLResponse(
        _health_html_page(
            out,
            model_path=STATE.model_path or "",
            max_new_tokens_effective=max_nt,
            page_generated_at=ts,
            cuda_available=cuda_ok,
            cuda_device_name=cuda_nm,
        )
    )
