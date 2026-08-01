from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.minimax.image_generation.transformation import (
    DEFAULT_API_BASE,
    MinimaxImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_is_registered():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="image-01",
        provider=LlmProviders.MINIMAX,
    )
    assert isinstance(config, MinimaxImageGenerationConfig)


def test_get_complete_url_supports_default_and_regional_base():
    config = MinimaxImageGenerationConfig()
    assert config.get_complete_url(None, None, "image-01", {}, {}) == DEFAULT_API_BASE
    assert (
        config.get_complete_url(
            "https://api.minimaxi.com/v1",
            None,
            "image-01",
            {},
            {},
        )
        == "https://api.minimaxi.com/v1/image_generation"
    )


def test_map_openai_params():
    config = MinimaxImageGenerationConfig()
    result = config.map_openai_params(
        non_default_params={
            "n": 2,
            "size": "1024x768",
            "response_format": "b64_json",
        },
        optional_params={},
        model="image-01",
        drop_params=False,
    )

    assert result == {
        "n": 2,
        "width": 1024,
        "height": 768,
        "response_format": "base64",
    }


def test_transform_request_flattens_extra_body():
    config = MinimaxImageGenerationConfig()
    result = config.transform_image_generation_request(
        model="image-01",
        prompt="A city skyline at sunrise",
        optional_params={
            "width": 1024,
            "height": 768,
            "subject_reference": "reference-data",
            "extra_body": {"width": 1536, "prompt_optimizer": True},
        },
        litellm_params={},
        headers={},
    )

    assert result == {
        "model": "image-01",
        "prompt": "A city skyline at sunrise",
        "width": 1536,
        "height": 768,
        "subject_reference": "reference-data",
        "prompt_optimizer": True,
    }


def test_image_generation_routes_request_and_parses_base64_response():
    captured_request = {}

    def capture_post(*args, **kwargs):
        captured_request.update(kwargs)
        return httpx.Response(
            status_code=200,
            json={
                "data": {"image_urls": ["encoded-image"]},
                "metadata": {"success_count": 1, "failed_count": 0},
                "base_resp": {"status_code": 0},
            },
            request=httpx.Request("POST", kwargs["url"]),
        )

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        side_effect=capture_post,
    ):
        response = litellm.image_generation(
            model="minimax/image-01",
            prompt="A city skyline at sunrise",
            size="1024x768",
            response_format="b64_json",
            n=1,
            subject_reference="reference-data",
            seed=42,
            extra_body={"prompt_optimizer": True},
            api_key="test-key",
        )

    assert captured_request["url"] == DEFAULT_API_BASE
    assert captured_request["headers"]["Authorization"] == "Bearer test-key"
    assert captured_request["json"] == {
        "model": "image-01",
        "prompt": "A city skyline at sunrise",
        "n": 1,
        "width": 1024,
        "height": 768,
        "response_format": "base64",
        "subject_reference": "reference-data",
        "seed": 42,
        "prompt_optimizer": True,
    }
    assert response.data[0].b64_json == "encoded-image"
    assert response.data[0].url is None
    assert response._hidden_params["metadata"] == {
        "success_count": 1,
        "failed_count": 0,
    }


def test_transform_image_generation_response_parses_urls():
    config = MinimaxImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "data": {"image_urls": ["https://example.com/generated.png"]},
            "metadata": {"success_count": 1, "failed_count": 0},
            "base_resp": {"status_code": 0},
        },
    )

    response = config.transform_image_generation_response(
        model="image-01-live",
        raw_response=raw_response,
        model_response=litellm.ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"response_format": "url"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.data[0].url == "https://example.com/generated.png"
    assert response.data[0].b64_json is None


@pytest.mark.parametrize(
    "status_code,response_body",
    [
        (400, {"base_resp": {"status_code": 1002, "status_msg": "Invalid request"}}),
        (200, {"base_resp": {"status_code": 1002, "status_msg": "Invalid request"}}),
    ],
)
def test_transform_image_generation_response_raises_for_errors(status_code, response_body):
    config = MinimaxImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=status_code,
        json=response_body,
        request=httpx.Request("POST", DEFAULT_API_BASE),
    )

    with pytest.raises(Exception, match="Invalid request"):
        config.transform_image_generation_response(
            model="image-01",
            raw_response=raw_response,
            model_response=litellm.ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
