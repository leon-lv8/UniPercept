from fastapi import APIRouter

from ..runtime.model_catalog import _model_catalog_entries
from ..runtime.state import _raise_if_model_unavailable

router = APIRouter()


@router.get("/v1/models")
async def list_models():
    _raise_if_model_unavailable()
    return {"object": "list", "data": _model_catalog_entries()}
