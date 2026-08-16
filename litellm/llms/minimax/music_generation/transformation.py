"""
MiniMax music generation transformation

Maps the OpenAI audio speech spec onto MiniMax `POST /v1/music_generation`
Reference: https://platform.minimax.io/docs/api-reference/music-generation
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.llms.openai import HttpxBinaryResponseContent

MUSIC_GENERATION_MODELS: Final = frozenset(("music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"))

MUSIC_REQUEST_FIELDS: Final = frozenset(
    (
        "lyrics",
        "stream",
        "output_format",
        "audio_setting",
        "lyrics_optimizer",
        "is_instrumental",
        "audio_url",
        "audio_base64",
        "cover_feature_id",
    )
)

_BODY_FIELDS: Final = MUSIC_REQUEST_FIELDS - frozenset(("stream", "output_format", "audio_setting"))
_AUDIO_SETTING_FIELDS: Final = frozenset(("sample_rate", "bitrate", "format"))
_AUDIO_FORMATS: Final = frozenset(("mp3", "wav", "pcm"))
_OUTPUT_FORMATS: Final = frozenset(("url", "hex"))
_GENERATION_COMPLETED_STATUS: Final = 2
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


class MinimaxMusicGenerationConfig(BaseTextToSpeechConfig):
    """
    Configuration for MiniMax music generation

    The global endpoint is `https://api.minimax.io/v1/music_generation` and the China endpoint is
    `https://api.minimaxi.com/v1/music_generation`, selected through `api_base` or `MINIMAX_API_BASE`.
    `input` carries the style prompt; vocal tracks additionally need `lyrics`
    """

    DEFAULT_API_BASE: Final = "https://api.minimax.io"
    ENDPOINT_PATH: Final = "/v1/music_generation"
    DEFAULT_MODEL: Final = "music-3.0"

    def __init__(self, audio_client: "HTTPHandler | None" = None) -> None:
        super().__init__()
        self._audio_client: Final = audio_client

    @staticmethod
    def is_music_model(model: str) -> bool:
        return model in MUSIC_GENERATION_MODELS

    def get_supported_openai_params(self, model: str) -> list:
        return ["response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: "str | Mapping[str, object] | None" = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict]:
        """
        Music generation has no voice selection, so the mapped voice is always None.

        `response_format` becomes the MiniMax audio format, and the documented music request fields are read
        from the call kwargs (directly or through `extra_body`) because `speech()` does not model them.
        """
        call_kwargs: Final = kwargs or _NO_PARAMS
        extra_body: Final = call_kwargs.get("extra_body")
        audio_format: Final = self._map_response_format(
            model=model,
            response_format=(optional_params or _NO_PARAMS).get("response_format"),
            drop_params=drop_params,
        )
        return None, {
            **self._music_fields(call_kwargs),
            **self._music_fields(extra_body if isinstance(extra_body, Mapping) else _NO_PARAMS),
            **({"format": audio_format} if audio_format else _NO_PARAMS),
        }

    @staticmethod
    def _music_fields(params: Mapping[str, object]) -> Mapping[str, object]:
        return {key: value for key, value in params.items() if key in MUSIC_REQUEST_FIELDS and value is not None}

    @staticmethod
    def _map_response_format(model: str, response_format: object, drop_params: bool) -> str | None:
        if response_format is None:
            return None
        if isinstance(response_format, str) and response_format.lower() in _AUDIO_FORMATS:
            return response_format.lower()
        if drop_params:
            return None
        raise litellm.UnsupportedParamsError(
            message=(
                f"MiniMax music generation supports response_format={sorted(_AUDIO_FORMATS)}, "
                f"got {response_format!r}. Pass drop_params=True to ignore it."
            ),
            model=model,
            llm_provider="minimax",
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key: Final = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if resolved_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )
        return {
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        """
        Accepts both `https://api.minimax.io` and `https://api.minimax.io/v1` style bases.
        """
        base_url: Final = (api_base or get_secret_str("MINIMAX_API_BASE") or self.DEFAULT_API_BASE).rstrip("/")
        return f"{base_url.removesuffix('/v1')}{self.ENDPOINT_PATH}"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | Headers
    ) -> BaseLLMException:
        return MinimaxException(message=error_message, status_code=status_code, headers=Headers(headers))

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        params: Final = optional_params or _NO_PARAMS

        if params.get("stream"):
            raise litellm.UnsupportedParamsError(
                message="MiniMax music generation streaming is not supported through litellm.speech().",
                model=model,
                llm_provider="minimax",
            )

        output_format: Final = str(params.get("output_format", "hex")).lower()
        if output_format not in _OUTPUT_FORMATS:
            raise litellm.UnsupportedParamsError(
                message=(
                    f"MiniMax music generation supports output_format={sorted(_OUTPUT_FORMATS)}, "
                    f"got {params.get('output_format')!r}."
                ),
                model=model,
                llm_provider="minimax",
            )

        audio_setting: Final = self._audio_setting(params)

        return TextToSpeechRequestData(
            dict_body={
                "model": model,
                "prompt": input,
                "output_format": output_format,
                **({"audio_setting": audio_setting} if audio_setting else _NO_PARAMS),
                **{key: value for key, value in params.items() if key in _BODY_FIELDS and value is not None},
            }
        )

    @staticmethod
    def _audio_setting(params: Mapping[str, object]) -> Mapping[str, object]:
        requested: Final = params.get("audio_setting")
        audio_format: Final = params.get("format")
        source: Final = requested if isinstance(requested, Mapping) else _NO_PARAMS
        merged: Final = {**source, "format": audio_format} if isinstance(audio_format, str) else source
        return {key: value for key, value in merged.items() if key in _AUDIO_SETTING_FIELDS and value is not None}

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        MiniMax returns `data.audio` as a hex string, or as a 24h download URL when `output_format='url'`.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        response_json: Final = self._parse_json(raw_response)
        self._raise_for_api_error(response_json=response_json, raw_response=raw_response)

        data: Final = response_json.get("data")
        if not isinstance(data, Mapping):
            raise self._error(f"response has no data object: {sorted(response_json)}", raw_response)

        status: Final = data.get("status")
        if status is not None and status != _GENERATION_COMPLETED_STATUS:
            raise self._error(f"generation did not complete, data.status={status}", raw_response)

        audio: Final = data.get("audio")
        if not isinstance(audio, str) or not audio:
            raise self._error("response has no audio payload", raw_response)

        if audio.startswith(("http://", "https://")):
            return HttpxBinaryResponseContent(self._download_audio(audio_url=audio, raw_response=raw_response))

        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                content=self._decode_hex_audio(audio=audio, raw_response=raw_response),
                request=raw_response.request,
            )
        )

    @staticmethod
    def _error(message: str, raw_response: httpx.Response, status_code: int | None = None) -> MinimaxException:
        return MinimaxException(
            status_code=status_code or raw_response.status_code,
            message=f"MiniMax music generation {message}",
            headers=raw_response.headers,
        )

    def _parse_json(self, raw_response: httpx.Response) -> Mapping[str, object]:
        try:
            parsed: Final = raw_response.json()
        except ValueError as e:
            raise self._error(f"response failed to parse as JSON: {e}", raw_response)
        if not isinstance(parsed, Mapping):
            raise self._error(f"response has unexpected type {type(parsed).__name__}", raw_response)
        return parsed

    def _raise_for_api_error(self, response_json: Mapping[str, object], raw_response: httpx.Response) -> None:
        base_resp: Final = response_json.get("base_resp")
        if not isinstance(base_resp, Mapping):
            return
        status_code: Final = base_resp.get("status_code")
        if status_code is None or status_code == 0:
            return
        raise self._error(
            f"failed with status_code {status_code}: {base_resp.get('status_msg', 'unknown error')}",
            raw_response,
        )

    def _decode_hex_audio(self, audio: str, raw_response: httpx.Response) -> bytes:
        try:
            return bytes.fromhex(audio)
        except ValueError as e:
            raise self._error(f"audio failed to decode as hex: {e}", raw_response)

    def _download_audio(self, audio_url: str, raw_response: httpx.Response) -> httpx.Response:
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client: Final = self._audio_client or _get_httpx_client()
        audio_response: Final = client.get(url=audio_url)
        if audio_response.status_code >= 400:
            raise self._error(
                f"audio failed to download from {audio_url}",
                raw_response,
                status_code=audio_response.status_code,
            )
        return audio_response
