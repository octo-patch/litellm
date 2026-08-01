from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast  # noqa: TID251, RUF100  # validated JSON boundaries require narrowing

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
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


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: Base image interface requires a list
        return ["n", "response_format", "size"]  # mutable-ok: Base image interface requires a list

    @staticmethod
    def _parse_size(size: object) -> tuple[int, int]:
        if not isinstance(size, str):
            raise TypeError("size must use the '<width>x<height>' format")
        dimensions = size.lower().split("x")
        if len(dimensions) != 2:
            raise ValueError("size must use the '<width>x<height>' format")
        try:
            width, height = (int(dimension) for dimension in dimensions)
        except ValueError as exc:
            raise ValueError("size must use the '<width>x<height>' format") from exc
        if width <= 0 or height <= 0:
            raise ValueError("size dimensions must be positive integers")
        return width, height

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: Base image interface returns mutable request parameters
        mapped_params = dict(optional_params)  # mutable-ok: Base image interface returns mutable request parameters
        supported_params = self.get_supported_openai_params(model)

        for key, value in non_default_params.items():
            if key not in supported_params:
                if drop_params:
                    continue
                raise ValueError(f"Parameter {key} is not supported for model {model}")
            if key == "size":
                width, height = self._parse_size(value)
                mapped_params["width"] = width
                mapped_params["height"] = height
            elif key == "response_format":
                if value == "b64_json":
                    mapped_params[key] = "base64"
                elif value in ("url", "base64"):
                    mapped_params[key] = value
                else:
                    raise ValueError("response_format must be 'url', 'b64_json', or 'base64'")
            else:
                mapped_params[key] = value

        return mapped_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url = api_base or get_secret_str("MINIMAX_API_BASE")
        if not base_url:
            return DEFAULT_API_BASE
        base_url = base_url.rstrip("/")
        if base_url.endswith("/image_generation"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/image_generation"
        return f"{base_url}/v1/image_generation"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: HTTP handlers require mutable headers
        final_api_key = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
        if not final_api_key:
            raise ValueError("MINIMAX_API_KEY is not set")
        return {  # mutable-ok: HTTP handlers require mutable headers
            **headers,
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        normalized_headers = (
            headers
            if isinstance(headers, httpx.Headers)
            else dict(headers)  # mutable-ok: Base exception requires concrete headers
        )
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=normalized_headers,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: Base image interface returns a mutable request body
        request_data: dict[str, object] = {  # mutable-ok: HTTP handler requires a mutable request body
            "model": model,
            "prompt": prompt,
        }
        extra_body = optional_params.get("extra_body")
        request_data.update(
            (key, value)
            for key, value in optional_params.items()
            if not key.startswith("_") and key not in ("extra_body", "extra_headers")
        )
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise ValueError("extra_body must be a dictionary")
            typed_extra_body = cast(  # cast-ok: validated dictionary
                Mapping[str, object], extra_body
            )
            request_data.update(typed_extra_body)
        return request_data

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
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
            response_value: object = raw_response.json()  # pyright: ignore[reportAny]  # validated below
        except ValueError as exc:
            raise self.get_error_class(
                error_message=f"Failed to parse image generation response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not isinstance(response_value, dict):
            raise self.get_error_class(
                error_message="Image generation response was not a JSON object",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        response_data = cast(  # cast-ok: validated JSON object
            Mapping[str, object], response_value
        )

        provider_status_code: object = None
        provider_message = "Unknown error"
        base_response_value = response_data.get("base_resp")
        if isinstance(base_response_value, dict):
            base_response = cast(  # cast-ok: validated JSON object
                Mapping[str, object], base_response_value
            )
            provider_status_code = base_response.get("status_code")
            provider_message_value = base_response.get("status_msg")
            if isinstance(provider_message_value, str):
                provider_message = provider_message_value
        if provider_status_code not in (None, 0, "0"):
            raise self.get_error_class(
                error_message=f"Image generation failed: {provider_message}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        response_images_value = response_data.get("data")
        if not isinstance(response_images_value, dict):
            raise self.get_error_class(
                error_message="Image generation response did not contain image data",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        response_images = cast(  # cast-ok: validated JSON object
            Mapping[str, object], response_images_value
        )
        image_urls_value = response_images.get("image_urls")
        if not isinstance(image_urls_value, list):
            raise self.get_error_class(
                error_message="Image generation response did not contain an image list",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        image_urls = cast(  # cast-ok: validated image list
            Sequence[object], image_urls_value
        )

        response_format = request_data.get("response_format", "url")
        if model_response.data is None:
            model_response.data = []  # mutable-ok: Image response requires a mutable data list
        for image_data in image_urls:
            if not isinstance(image_data, str):
                continue
            if response_format in ("base64", "b64_json"):
                model_response.data.append(ImageObject(b64_json=image_data))
            else:
                model_response.data.append(ImageObject(url=image_data))

        metadata_value = response_data.get("metadata")
        if isinstance(metadata_value, dict):
            metadata = cast(  # cast-ok: validated JSON object
                Mapping[str, object], metadata_value
            )
            model_response._hidden_params["metadata"] = metadata  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # response metadata contract

        return model_response
