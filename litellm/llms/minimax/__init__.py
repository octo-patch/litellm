"""
MiniMax LLM Provider
"""

from .text_to_speech.transformation import (
    MinimaxException,
    MinimaxTextToSpeechConfig,
)
from .voice_clone.transformation import MinimaxVoiceCloneConfig

__all__ = [
    "MinimaxException",
    "MinimaxTextToSpeechConfig",
    "MinimaxVoiceCloneConfig",
]
