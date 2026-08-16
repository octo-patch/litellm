"""
MiniMax LLM Provider
"""

from .music_generation.transformation import MinimaxMusicGenerationConfig
from .text_to_speech.transformation import (
    MinimaxException,
    MinimaxTextToSpeechConfig,
)

__all__ = [
    "MinimaxException",
    "MinimaxMusicGenerationConfig",
    "MinimaxTextToSpeechConfig",
]
