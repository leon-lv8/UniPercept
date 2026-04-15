from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class ChatMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Any]]
    name: str


class ChatCompletionRequest(TypedDict, total=False):
    model: str
    messages: List[ChatMessage]
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool


class ChatCompletionResponse(TypedDict, total=False):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatCompletionChunk(TypedDict, total=False):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelCapabilities(TypedDict, total=False):
    """Optional extensions on /v1/models entries for capability-aware clients."""

    multimodal: bool
    vision: bool
    image_input: bool
    text_output: bool


class ModelObject(TypedDict, total=False):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    model_type: str
    modalities: Dict[str, List[str]]
    capabilities: ModelCapabilities
    supported_input_modalities: List[str]
    supported_output_modalities: List[str]

