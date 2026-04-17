from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from .env_utils import _select_device
from .model_load import _load_model_worker
from .state import STATE

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.model_id = os.environ.get("MODEL_ID", "unipercept")
    STATE.model_path = os.environ.get("MODEL_PATH", "/models/unipercept")
    STATE.device = _select_device()

    if not os.path.exists(STATE.model_path):
        raise RuntimeError(f"MODEL_PATH does not exist: {STATE.model_path}")

    STATE.model = None
    STATE.tokenizer = None
    STATE.gen_cfg = None
    STATE.model_list_created = 0
    STATE.model_load_error = None
    STATE.model_loading = True
    app.state.load_seconds = None

    async def _run_load() -> None:
        try:
            await asyncio.to_thread(_load_model_worker, app)
        except Exception:
            logger.exception("Background model load task raised")
            if STATE.model_load_error is None:
                STATE.model_load_error = "后台加载任务异常退出（详见服务日志）。"
            STATE.model_loading = False

    app.state.model_load_task = asyncio.create_task(_run_load())

    yield

    t = getattr(app.state, "model_load_task", None)
    if t is not None and not t.done():
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t

