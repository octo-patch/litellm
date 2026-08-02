from typing import TYPE_CHECKING, Any

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_OPENAI_SIZE_TO_ASPECT_RATIO = {
    "256x256": "1:1",
    "512x512": "1:1",
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1792x1024": "16:9",
    "1024x1536": "2:3",
    "1024x1792": "9:16",
    "1:1": "1:1",
    "3:2": "3:2",
    "2:3": "2:3",
    "16:9": "16:9",
    "9:16": "9:16",
}


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "response_format",
            "size",
            "aspect_ratio",
            "width",
            "height",
            "seed",
            "prompt_optimizer",
            "subject_reference",
            "extra_body",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for key, value in non_default_params.items():
            if key == "size":
                optional_params["aspect_ratio"] = _OPENAI_SIZE_TO_ASPECT_RATIO.get(str(value), str(value))
            elif key == "response_format" and value == "b64_json":
                optional_params[key] = "base64"
            elif key == "extra_body" and isinstance(value, dict):
                optional_params.update({k: v for k, v in value.items() if v is not None})
            else:
                optional_params[key] = value

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key = api_key or get_secret_str("MINIMAX_API_KEY")
        if not final_api_key:
            raise ValueError(
                "MINIMAX_API_KEY is not set. Please set it via environment variable or pass api_key parameter."
            )

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/v1"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/image_generation"):
            return base_url
        return f"{base_url}/image_generation"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        extra_body = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            request_body.update({k: v for k, v in extra_body.items() if v is not None})

        for key, value in optional_params.items():
            if key in {"extra_body", "size"} or value is None:
                continue
            if key == "response_format" and value == "b64_json":
                request_body[key] = "base64"
                continue
            request_body[key] = value

        if "size" in optional_params and "aspect_ratio" not in request_body:
            request_body["aspect_ratio"] = _OPENAI_SIZE_TO_ASPECT_RATIO.get(
                str(optional_params["size"]),
                str(optional_params["size"]),
            )

        return request_body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Error parsing MiniMax image response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        base_resp = response_data.get("base_resp", {})
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            raise self.get_error_class(
                error_message=f"MiniMax image generation failed: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_data,
        )

        image_urls = []
        data_block = response_data.get("data", {})
        if isinstance(data_block, dict):
            image_urls = data_block.get("image_urls", []) or []
        elif isinstance(data_block, list):
            image_urls = data_block

        if not model_response.data:
            model_response.data = []

        response_format = request_data.get("response_format")
        for item in image_urls:
            if isinstance(item, dict):
                url = item.get("url") or item.get("image_url")
                b64_json = item.get("b64_json") or item.get("base64")
            else:
                url = item
                b64_json = None

            if isinstance(url, str) and url.startswith("data:"):
                _, encoded = url.split(",", 1) if "," in url else (url, "")
                b64_json = encoded or b64_json
                url = None
            elif isinstance(url, str) and url.startswith(("http://", "https://")):
                pass
            elif isinstance(url, str):
                if response_format in {"base64", "b64_json"}:
                    b64_json = url
                    url = None
                elif b64_json is None:
                    b64_json = url
                    url = None

            model_response.data.append(
                ImageObject(
                    url=url,
                    b64_json=b64_json,
                    revised_prompt=None,
                )
            )

        model_response._hidden_params["model"] = response_data.get("model", model)
        model_response._hidden_params["metadata"] = response_data.get("metadata", {})
        model_response._hidden_params["base_resp"] = base_resp
        return model_response
