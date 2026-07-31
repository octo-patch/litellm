import httpx
import litellm
import pytest
from unittest.mock import patch

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.minimax.image_generation.transformation import (
    MinimaxImageGenerationConfig,
)
from litellm.types.utils import ImageResponse
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders


class TestMinimaxImageGenerationTransformation:
    def setup_method(self):
        self.config = MinimaxImageGenerationConfig()

    def test_provider_config_manager_returns_minimax_config(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model="image-01",
            provider=LlmProviders.MINIMAX,
        )
        assert isinstance(config, MinimaxImageGenerationConfig)

    @pytest.mark.parametrize(
        "api_base,expected_url",
        [
            ("https://api.minimax.io", "https://api.minimax.io/v1/image_generation"),
            ("https://api.minimaxi.com/v1", "https://api.minimaxi.com/v1/image_generation"),
            (
                "https://api.minimax.io/v1/image_generation",
                "https://api.minimax.io/v1/image_generation",
            ),
        ],
    )
    def test_get_complete_url(self, api_base, expected_url):
        assert (
            self.config.get_complete_url(
                api_base=api_base,
                api_key=None,
                model="image-01",
                optional_params={},
                litellm_params={},
            )
            == expected_url
        )

    def test_validate_environment_uses_litellm_api_key_param(self):
        headers = self.config.validate_environment(
            headers={},
            model="image-01",
            messages=[],
            optional_params={},
            litellm_params={"api_key": "test-key"},
        )
        assert headers == {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }

    def test_transform_request_forwards_minimax_fields(self):
        optional_params = self.config.map_openai_params(
            non_default_params={"n": 2, "response_format": "b64_json"},
            optional_params={},
            model="image-01",
            drop_params=False,
        )
        optional_params.update(
            {
                "subject_reference": [{"type": "character", "image_file": "https://example.com/ref.jpg"}],
                "aspect_ratio": "16:9",
                "width": 1280,
                "height": 720,
                "seed": 7,
                "prompt_optimizer": True,
            }
        )

        request = self.config.transform_image_generation_request(
            model="image-01",
            prompt="A city at sunset",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert request == {
            "model": "image-01",
            "prompt": "A city at sunset",
            "subject_reference": [{"type": "character", "image_file": "https://example.com/ref.jpg"}],
            "aspect_ratio": "16:9",
            "width": 1280,
            "height": 720,
            "response_format": "base64",
            "seed": 7,
            "n": 2,
            "prompt_optimizer": True,
        }

    def test_transform_url_response_preserves_metadata(self):
        response = httpx.Response(
            200,
            json={
                "id": "request-id",
                "data": {"image_urls": ["https://example.com/image.png"]},
                "metadata": {"success_count": 1, "failed_count": 0},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
            request=httpx.Request("POST", "https://api.minimax.io/v1/image_generation"),
        )
        result = self.config.transform_image_generation_response(
            model="image-01",
            raw_response=response,
            model_response=ImageResponse(data=[]),
            logging_obj=None,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.data[0].url == "https://example.com/image.png"
        assert result._hidden_params["minimax_request_id"] == "request-id"
        assert result._hidden_params["minimax_metadata"]["success_count"] == 1

    def test_transform_base64_response(self):
        response = httpx.Response(
            200,
            json={
                "data": {"image_base64": ["encoded-image"]},
                "base_resp": {"status_code": 0},
            },
            request=httpx.Request("POST", "https://api.minimax.io/v1/image_generation"),
        )
        result = self.config.transform_image_generation_response(
            model="image-01",
            raw_response=response,
            model_response=ImageResponse(data=[]),
            logging_obj=None,
            request_data={},
            optional_params={"response_format": "base64"},
            litellm_params={},
            encoding=None,
        )

        assert result.data[0].b64_json == "encoded-image"
        assert result.data[0].url is None

    def test_transform_api_error(self):
        response = httpx.Response(
            200,
            json={"base_resp": {"status_code": 2013, "status_msg": "invalid input"}},
            request=httpx.Request("POST", "https://api.minimax.io/v1/image_generation"),
        )
        with pytest.raises(BaseLLMException, match="invalid input"):
            self.config.transform_image_generation_response(
                model="image-01",
                raw_response=response,
                model_response=ImageResponse(data=[]),
                logging_obj=None,
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_public_image_generation_routes_to_minimax(self, mock_post):
        mock_post.return_value = httpx.Response(
            200,
            json={
                "data": {"image_urls": ["https://example.com/image.png"]},
                "base_resp": {"status_code": 0},
            },
            request=httpx.Request("POST", "https://api.minimax.io/v1/image_generation"),
        )

        result = litellm.image_generation(
            model="minimax/image-01",
            prompt="A city at sunset",
            api_key="test-key",
            api_base="https://api.minimax.io/v1",
            n=1,
            response_format="url",
            aspect_ratio="16:9",
            seed=7,
            prompt_optimizer=True,
        )

        assert result.data[0].url == "https://example.com/image.png"
        request_kwargs = mock_post.call_args.kwargs
        assert request_kwargs["url"] == "https://api.minimax.io/v1/image_generation"
        assert request_kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert request_kwargs["json"] == {
            "model": "image-01",
            "prompt": "A city at sunset",
            "n": 1,
            "response_format": "url",
            "aspect_ratio": "16:9",
            "seed": 7,
            "prompt_optimizer": True,
        }
