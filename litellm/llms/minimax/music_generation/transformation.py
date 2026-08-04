"""
MiniMax Music Generation transformation

Maps music generation requests to the MiniMax Music Generation API.
Reference: https://platform.minimax.io/docs/api-reference/music-generation
"""

from typing import TYPE_CHECKING, Any, ClassVar

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class MinimaxMusicGenerationException(BaseLLMException):
    """Custom exception for MiniMax Music Generation API errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


class MinimaxMusicGenerationConfig(BaseTextToSpeechConfig):
    """
    Configuration for MiniMax Music Generation

    Reference: https://platform.minimax.io/docs/api-reference/music-generation

    MiniMax exposes POST {base}/v1/music_generation which accepts a JSON body
    with the model and a prompt (and optional lyrics, stream, output_format,
    audio_setting, lyrics_optimizer, is_instrumental, audio_url, audio_base64,
    cover_feature_id) and returns:

        {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"status": 2, "audio": "<audio>"}
        }

    `data.audio` is either a signed URL (output_format="url", 24h TTL) or
    hex-encoded audio bytes (output_format="hex").
    """

    MUSIC_BASE_URL = "https://api.minimax.io"
    MUSIC_ENDPOINT_PATH = "/music_generation"

    # Request fields accepted by the MiniMax music generation API
    SUPPORTED_REQUEST_FIELDS: ClassVar[list[str]] = [
        "lyrics",
        "stream",
        "output_format",
        "audio_setting",
        "lyrics_optimizer",
        "is_instrumental",
        "audio_url",
        "audio_base64",
        "cover_feature_id",
    ]

    def get_supported_openai_params(self, model: str) -> list:
        """
        MiniMax music generation accepts these provider-specific parameters.
        """
        return list(self.SUPPORTED_REQUEST_FIELDS)

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict]:
        """
        MiniMax music generation has no voice mapping; pass provider-specific
        parameters through unchanged.
        """
        return None, dict(optional_params) if optional_params else {}

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate MiniMax environment and set up authentication headers.
        """
        api_key = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")

        if api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> BaseLLMException:
        return MinimaxMusicGenerationException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Build the MiniMax music generation request payload.

        - model: the MiniMax music model (e.g. music-3.0)
        - prompt: the music description
        """
        params = dict(optional_params) if optional_params else {}

        request_body: dict[str, Any] = {
            "model": model,
            "prompt": input,
        }

        for key in self.SUPPORTED_REQUEST_FIELDS:
            if key in params and params[key] is not None:
                request_body[key] = params[key]

        # Pass through any additional MiniMax-specific parameters
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None and key not in request_body:
                    request_body[key] = value

        return TextToSpeechRequestData(
            dict_body=request_body,
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        """
        Transform the MiniMax music generation response to standard audio bytes.
        """
        import json

        from litellm.llms.custom_httpx.http_handler import _get_httpx_client
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            response_json = raw_response.json()
        except json.JSONDecodeError as e:
            raise MinimaxMusicGenerationException(
                status_code=500,
                message=f"Failed to parse MiniMax music generation response: {e}",
                headers=dict(raw_response.headers),
            )

        # MiniMax base response status check
        base_resp = response_json.get("base_resp", {}) or {}
        if not isinstance(base_resp, dict):
            base_resp = {}
        if base_resp.get("status_code", 0) != 0:
            status_msg = base_resp.get("status_msg", "Unknown error")
            raise MinimaxMusicGenerationException(
                status_code=raw_response.status_code,
                message=f"MiniMax music generation error: {status_msg}",
                headers=dict(raw_response.headers),
            )

        data = response_json.get("data", {}) or {}
        if not isinstance(data, dict):
            raise MinimaxMusicGenerationException(
                status_code=500,
                message=(
                    f"MiniMax music generation returned an unexpected data payload: "
                    f"{str(response_json)[:200]}"
                ),
                headers=dict(raw_response.headers),
            )

        status = data.get("status")
        audio = data.get("audio")
        if not audio and status == 1:
            raise MinimaxMusicGenerationException(
                status_code=500,
                message=(
                    "MiniMax music generation is still in progress (status=1) and "
                    "returned no audio. Retry the request."
                ),
                headers=dict(raw_response.headers),
            )
        if not audio:
            raise MinimaxMusicGenerationException(
                status_code=500,
                message=(
                    f"No audio in MiniMax music generation response. Response keys: "
                    f"{list(response_json.keys())}"
                ),
                headers=dict(raw_response.headers),
            )

        # MiniMax returns the audio as a URL (output_format="url") or hex (output_format="hex")
        audio_bytes: bytes
        if isinstance(audio, str) and audio.startswith(("http://", "https://")):
            audio_resp = _get_httpx_client().get(url=audio)
            audio_resp.raise_for_status()
            audio_bytes = audio_resp.content
        else:
            try:
                audio_bytes = bytes.fromhex(audio)
            except (TypeError, ValueError) as e:
                raise MinimaxMusicGenerationException(
                    status_code=500,
                    message=f"Failed to decode MiniMax music generation audio: {e}",
                    headers=dict(raw_response.headers),
                )

        clean_headers = dict(raw_response.headers)
        clean_headers.pop("content-encoding", None)
        clean_headers.pop("transfer-encoding", None)
        clean_headers["content-length"] = str(len(audio_bytes))

        binary_response = httpx.Response(
            status_code=200,
            headers=clean_headers,
            content=audio_bytes,
            request=raw_response.request,
        )
        return HttpxBinaryResponseContent(binary_response)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Construct the MiniMax music generation endpoint URL.

        Handles api_base values with and without a trailing `/v1` so both the
        global (`https://api.minimax.io`) and CN (`https://api.minimaxi.com`)
        regional endpoints resolve to `{base}/v1/music_generation`.
        """
        base_url = api_base or get_secret_str("MINIMAX_API_BASE") or self.MUSIC_BASE_URL
        base_url = base_url.rstrip("/")

        # Ensure it ends with /v1/music_generation
        if base_url.endswith(self.MUSIC_ENDPOINT_PATH):
            return base_url
        elif base_url.endswith("/v1"):
            return f"{base_url}{self.MUSIC_ENDPOINT_PATH}"
        else:
            return f"{base_url}/v1{self.MUSIC_ENDPOINT_PATH}"
