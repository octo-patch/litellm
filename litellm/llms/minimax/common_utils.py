from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

MINIMAX_API_HOST: Final = "https://api.minimax.io"
_API_VERSION_SUFFIX: Final = "/v1"


class MinimaxException(BaseLLMException):
    pass


def resolve_minimax_api_host(api_base: str | None = None) -> str:
    """MINIMAX_API_BASE is documented with the version suffix, so trim it back to the bare host."""
    base: Final = (api_base or get_secret_str("MINIMAX_API_BASE") or MINIMAX_API_HOST).rstrip("/")
    return base.removesuffix(_API_VERSION_SUFFIX)
