from typing import TYPE_CHECKING

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


DEFAULT_API_BASE = "https://api.minimax.io/v1/image_generation"
IMAGE_GENERATION_PATH = "/v1/image_generation"
IMAGE_REQUEST_FIELDS = (
    "subject_reference",
    "aspect_ratio",
    "width",
    "height",
    "response_format",
    "seed",
    "n",
    "prompt_optimizer",
)


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    """Configuration for MiniMax image generation."""

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: required by BaseImageGenerationConfig.
        return ["n", "response_format"]  # mutable-ok: required by LiteLLM's parameter contract.

    @staticmethod
    def _normalize_response_format(value: object) -> object:
        if value == "b64_json":
            return "base64"
        return value

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        optional_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: required by BaseImageGenerationConfig.
        supported_params = self.get_supported_openai_params(model)
        for key, value in non_default_params.items():
            if key in optional_params:
                continue
            if key not in supported_params:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    "Set drop_params=True to drop unsupported parameters."
                )
            optional_params[key] = self._normalize_response_format(value)
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        litellm_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        stream: bool | None = None,
    ) -> str:
        base_url = (
            api_base or litellm_params.get("api_base") or get_secret_str("MINIMAX_API_BASE") or DEFAULT_API_BASE
        ).rstrip("/")
        if base_url.endswith(IMAGE_GENERATION_PATH):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/image_generation"
        return f"{base_url}{IMAGE_GENERATION_PATH}"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: required by BaseImageGenerationConfig.
        optional_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        litellm_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: required by BaseImageGenerationConfig.
        final_api_key = api_key or litellm_params.get("api_key") or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
        if not final_api_key:
            raise ValueError("MINIMAX_API_KEY is not set")
        headers.update(
            {  # mutable-ok: headers are intentionally mutated by the shared HTTP handler.
                "Authorization": f"Bearer {final_api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        litellm_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        headers: dict,  # mutable-ok: required by BaseImageGenerationConfig.
    ) -> dict:  # mutable-ok: required by BaseImageGenerationConfig.
        request_data = {  # mutable-ok: request payload is built for the provider API.
            "model": model,
            "prompt": prompt,
        }
        for key in IMAGE_REQUEST_FIELDS:
            value = optional_params.get(key)
            if value is not None:
                request_data[key] = self._normalize_response_format(value) if key == "response_format" else value
        return request_data

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        optional_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        litellm_params: dict,  # mutable-ok: required by BaseImageGenerationConfig.
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Failed to parse MiniMax image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        base_resp = response_data.get("base_resp") or {}  # mutable-ok: provider JSON is normalized into a mapping.
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            raise self.get_error_class(
                error_message=str(base_resp.get("status_msg") or base_resp),
                status_code=400,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []  # mutable-ok: LiteLLM accumulates images in a mutable response list.

        data = response_data.get("data") or {}  # mutable-ok: provider JSON is normalized into a mapping.
        for image_url in data.get("image_urls") or ():
            if isinstance(image_url, str):
                model_response.data.append(ImageObject(url=image_url))
        for image_base64 in data.get("image_base64") or ():
            if isinstance(image_base64, str):
                model_response.data.append(ImageObject(b64_json=image_base64))

        for key, value in (
            ("minimax_request_id", response_data.get("id")),
            ("minimax_metadata", response_data.get("metadata")),
            ("minimax_base_resp", base_resp),
        ):
            if value is not None:
                model_response._hidden_params[key] = value
        return model_response
