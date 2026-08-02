"""
Test MiniMax image generation support.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../"))

from litellm.llms.minimax.image_generation import (  # noqa: E402
    MinimaxImageGenerationConfig,
    get_minimax_image_generation_config,
)
from litellm.types.utils import ImageResponse, LlmProviders  # noqa: E402
from litellm.utils import ProviderConfigManager  # noqa: E402


def test_minimax_image_generation_config_factory():
    config = get_minimax_image_generation_config("image-01")
    assert isinstance(config, MinimaxImageGenerationConfig)


def test_minimax_provider_config_manager_returns_image_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="image-01",
        provider=LlmProviders.MINIMAX,
    )
    assert isinstance(config, MinimaxImageGenerationConfig)


def test_minimax_image_generation_request_and_response():
    config = MinimaxImageGenerationConfig()
    request = config.transform_image_generation_request(
        model="image-01",
        prompt="A red robot in a bright studio",
        optional_params={
            "n": 2,
            "response_format": "b64_json",
            "size": "1024x1024",
            "aspect_ratio": "16:9",
            "width": 1024,
            "height": 768,
            "seed": 7,
            "prompt_optimizer": True,
            "subject_reference": "https://example.com/reference.png",
        },
        litellm_params={},
        headers={},
    )

    assert request["model"] == "image-01"
    assert request["prompt"] == "A red robot in a bright studio"
    assert request["response_format"] == "base64"
    assert request["aspect_ratio"] == "16:9"
    assert request["width"] == 1024
    assert request["height"] == 768
    assert request["seed"] == 7
    assert request["prompt_optimizer"] is True
    assert request["subject_reference"] == "https://example.com/reference.png"

    response = MagicMock()
    response.json.return_value = {
        "data": {
            "image_urls": [
                "https://example.com/image-1.png",
                "data:image/png;base64,AAAA",
            ]
        },
        "metadata": {"success_count": 2, "failed_count": 0},
        "base_resp": {"status_code": 0},
        "model": "image-01",
    }
    response.status_code = 200
    response.headers = {}

    model_response = ImageResponse(data=[])
    logging_obj = MagicMock()

    result = config.transform_image_generation_response(
        model="image-01",
        raw_response=response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data=request,
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert len(result.data) == 2
    assert result.data[0].url == "https://example.com/image-1.png"
    assert result.data[1].b64_json == "AAAA"
    assert result._hidden_params["metadata"]["success_count"] == 2
    assert result._hidden_params["model"] == "image-01"
