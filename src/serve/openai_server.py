from __future__ import annotations

import sys

sys.path.append("src")

from .runtime.runtime_yaml import apply_runtime_yaml_to_environ

apply_runtime_yaml_to_environ()

from .runtime.env_utils import _configure_debug_logging_from_env

_configure_debug_logging_from_env()

# Emit a startup line via uvicorn logger so it's always visible in docker logs.
import os
import logging

logging.getLogger("uvicorn.error").info(
    "启动配置：RUNTIME_CONFIG_FILE=%s ENABLE_DEBUG_LOG=%s "
    "FORCE_RECLAIM_BETWEEN_TOWERS=%s WEIGHT_TOWER_LOAD_ORDER=%s",
    os.environ.get("RUNTIME_CONFIG_FILE"),
    os.environ.get("ENABLE_DEBUG_LOG"),
    os.environ.get("FORCE_RECLAIM_BETWEEN_TOWERS"),
    os.environ.get("WEIGHT_TOWER_LOAD_ORDER"),
)

from fastapi import FastAPI

from .runtime.lifespan import lifespan
from .routes.admin_prompt import router as admin_prompt_router
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.models import router as models_router

app = FastAPI(title="UniPercept OpenAI-Compatible Server", lifespan=lifespan)
app.include_router(health_router)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(admin_prompt_router)
