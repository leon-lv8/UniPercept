from __future__ import annotations

import os
from typing import List

from .env_utils import _env_bool, _now_ts
from ..openai_types import ModelObject
from .state import STATE


def _model_list_entry(model_id: str) -> ModelObject:
    created = STATE.model_list_created or _now_ts()
    owned_by = os.environ.get("MODEL_OWNED_BY", "unipercept")
    base: ModelObject = {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": owned_by,
    }
    if not _env_bool("MODEL_ADVERTISE_MULTIMODAL", True):
        base["model_type"] = "text"
        base["modalities"] = {"input": ["text"], "output": ["text"]}
        base["capabilities"] = {
            "multimodal": False,
            "vision": False,
            "image_input": False,
            "text_output": True,
        }
        base["supported_input_modalities"] = ["text"]
        base["supported_output_modalities"] = ["text"]
        return base

    base["model_type"] = "multimodal"
    base["modalities"] = {"input": ["text", "image"], "output": ["text"]}
    base["capabilities"] = {
        "multimodal": True,
        "vision": True,
        "image_input": True,
        "text_output": True,
    }
    base["supported_input_modalities"] = ["text", "image"]
    base["supported_output_modalities"] = ["text"]
    return base


def _model_catalog_with_extra_ids(entries: List[ModelObject]) -> List[ModelObject]:
    seen = {e["id"] for e in entries}
    out = list(entries)
    for raw in os.environ.get("MODEL_LIST_EXTRA_IDS", "").split(","):
        eid = raw.strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(_model_list_entry(eid))
    return out


def _model_catalog_entries() -> List[ModelObject]:
    primary_id = STATE.model_id
    primary = _model_list_entry(primary_id)
    return _model_catalog_with_extra_ids([primary])
