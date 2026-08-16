"""
MiniMax pass-through configuration

Routes MiniMax operations that the unified endpoints do not cover, such as the voice cloning audio
upload and voice clone endpoints.

Reference:
- https://platform.minimax.io/docs/api-reference/voice-cloning-clone
- https://platform.minimax.io/docs/api-reference/voice-cloning-uploadcloneaudio
- https://platform.minimaxi.com/docs/api-reference/voice-cloning-clone
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import litellm
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.minimax.common_utils import resolve_minimax_api_host
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from httpx import URL


class MinimaxPassthroughConfig(BasePassthroughConfig):
    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return []

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return resolve_minimax_api_host(api_base)

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return bool(request_data.get("stream", False))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: Mapping[str, object],
    ) -> tuple["URL", str]:
        base_target_url: Final = self.get_api_base(api_base)
        return (
            self.format_url(endpoint, base_target_url, request_query_params),
            base_target_url,
        )
