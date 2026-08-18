"""
MiniMax pass-through configuration.

Exposes the MiniMax operations that have no OpenAI-compatible route, such as the
voice clone flow ([docs](https://platform.minimax.io/docs/api-reference/voice-cloning-clone)):

- `POST v1/files/upload` - multipart `file` + `purpose` (`voice_clone` for the audio to
  clone, `prompt_audio` for the optional prompt), returns `file_id`
- `POST v1/voice_clone` - `file_id`, `voice_id` and `model`, returns the cloned
  `voice_id` alongside `base_resp.status_code`
- `POST v1/voice_design` - `prompt` and `voice_id`

The returned `voice_id` can then be used as the `voice` of a text-to-speech request.
"""

from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from httpx import URL

_VERSION_SUFFIX: Final = "/v1"


class MinimaxPassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict[str, object]) -> bool:
        return bool(request_data.get("stream", False))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict[str, object] | None,
        litellm_params: dict[str, object],
    ) -> tuple["URL", str]:
        """
        Build the target url from the provider's own endpoint path, e.g. `v1/voice_clone`.

        The configured api base already carries the version segment for the OpenAI
        compatible routes, so it is dropped here to keep the caller's documented path
        from being prefixed twice.
        """
        base_target_url: Final = self.get_api_base(api_base)

        if base_target_url is None:
            raise ValueError("MiniMax api base not found")

        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        api_key = self.get_api_key(api_key)

        if api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        # Content-Type is left to the caller: the audio upload is multipart, the clone
        # and design operations are JSON.
        headers["Authorization"] = f"Bearer {api_key}"

        return headers

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        base_target_url: Final = MinimaxChatConfig.get_api_base(api_base).rstrip("/")
        if base_target_url.endswith(_VERSION_SUFFIX):
            return base_target_url[: -len(_VERSION_SUFFIX)]
        return base_target_url

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return MinimaxChatConfig.get_api_key(api_key)

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return super().get_models(api_key, api_base)
