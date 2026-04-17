from __future__ import annotations

import sys

sys.path.append("src")

from .runtime.runtime_yaml import apply_runtime_yaml_to_environ

apply_runtime_yaml_to_environ()

from fastapi import FastAPI

from .runtime.lifespan import lifespan
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.models import router as models_router

app = FastAPI(title="UniPercept OpenAI-Compatible Server", lifespan=lifespan)
app.include_router(health_router)
app.include_router(models_router)
app.include_router(chat_router)
