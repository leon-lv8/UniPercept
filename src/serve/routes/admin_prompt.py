from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..runtime.model_load import _reload_system_prompt_runtime

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/admin/prompt/reload")
async def reload_system_prompt(req: Request):
    # Debug log keeps request-level breadcrumbs for ops troubleshooting.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("收到系统提示词重载请求：path=%s client=%s", str(req.url.path), getattr(req.client, "host", None))
    try:
        result = await _reload_system_prompt_runtime()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("系统提示词重载失败")
        raise HTTPException(status_code=500, detail=f"系统提示词重载失败：{e}") from e
    return JSONResponse(result)
