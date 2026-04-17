from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import torch
from fastapi import HTTPException

from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

logger = logging.getLogger(__name__)


class _State:
    model: Optional[InternVLChatModel] = None
    tokenizer: Optional[object] = None
    gen_cfg: Optional[Dict[str, Any]] = None
    device: Optional[torch.device] = None
    model_id: str = "unipercept"
    model_path: str = ""
    model_list_created: int = 0
    model_loading: bool = False
    model_load_error: Optional[str] = None
    lock: asyncio.Lock

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


STATE = _State()


def _model_unavailable_detail() -> str:
    if STATE.model_load_error:
        return f"模型加载失败：{STATE.model_load_error}"
    if STATE.model_loading:
        return "模型正在后台加载中，请稍后再试；也可访问 GET /health 查看 model_loaded、model_loading 等字段。"
    return "模型未就绪，无法处理推理请求。"


def _raise_if_model_unavailable() -> None:
    if (
        STATE.model is not None
        and STATE.tokenizer is not None
        and STATE.gen_cfg is not None
        and STATE.device is not None
    ):
        return
    raise HTTPException(status_code=503, detail=_model_unavailable_detail())
