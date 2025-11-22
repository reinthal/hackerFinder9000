"""OpenSafety AI data models."""

from src.models.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ErrorResponse,
    ErrorDetail,
    MessageRole,
    ModelInfo,
    ModelsResponse,
    StreamChoice,
    Tool,
    ToolCall,
    Usage,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatMessage",
    "Choice",
    "ChoiceMessage",
    "DeltaMessage",
    "ErrorResponse",
    "ErrorDetail",
    "MessageRole",
    "ModelInfo",
    "ModelsResponse",
    "StreamChoice",
    "Tool",
    "ToolCall",
    "Usage",
]
