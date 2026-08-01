from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import MinimaxImageGenerationConfig

__all__ = ("MinimaxImageGenerationConfig",)


def get_minimax_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return MinimaxImageGenerationConfig()
