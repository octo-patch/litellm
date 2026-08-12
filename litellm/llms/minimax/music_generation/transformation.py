from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.llms.base_llm.text_to_speech.transformation import (
    TextToSpeechRequestData,
)
from litellm.llms.minimax.text_to_speech.transformation import (
    MinimaxException,
    MinimaxTextToSpeechConfig,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class _MusicData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    audio: str
    status: int


class _MusicBaseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status_code: int
    status_msg: str = ""


class _MusicResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: _MusicData
    base_resp: _MusicBaseResponse


class _ParsedMusicResponse(BaseModel):
    audio: str
    status: int
    status_code: int
    status_message: str


class MinimaxMusicGenerationConfig(MinimaxTextToSpeechConfig):
    DEFAULT_BASE_URL: Final = "https://api.minimax.io"
    CHINA_BASE_URL: Final = "https://api.minimaxi.com"
    ENDPOINT_PATH: Final = "/v1/music_generation"
    SUPPORTED_MODELS: Final = frozenset(
        {
            "music-3.0",
            "music-2.6",
            "music-3.0-free",
            "music-2.6-free",
        }
    )
    _REQUEST_FIELDS: Final = frozenset(
        {
            "prompt",
            "lyrics",
            "stream",
            "output_format",
            "audio_setting",
            "lyrics_optimizer",
            "is_instrumental",
            "aigc_watermark",
        }
    )
    _AUDIO_FORMATS: Final = frozenset({"mp3", "wav", "pcm"})
    _OUTPUT_FORMATS: Final = frozenset({"url", "hex"})

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return model in cls.SUPPORTED_MODELS

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],
        voice: str | dict[str, object] | None = None,
        drop_params: bool = False,
        kwargs: dict[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object]]:
        extra_body_value: Final = (kwargs or {}).get("extra_body")
        extra_body: Final[dict[str, object]] = (
            cast(dict[str, object], extra_body_value) if isinstance(extra_body_value, dict) else {}
        )
        response_format_value: Final = optional_params.get("response_format", "mp3")
        response_format: Final = response_format_value if isinstance(response_format_value, str) else "mp3"
        audio_setting_value: Final = extra_body.get("audio_setting")
        audio_setting: Final[dict[str, object]] = (
            cast(dict[str, object], audio_setting_value) if isinstance(audio_setting_value, dict) else {}
        )
        audio_format_value: Final = audio_setting.get("format", response_format)
        if not isinstance(audio_format_value, str) or audio_format_value not in self._AUDIO_FORMATS:
            raise ValueError("MiniMax music audio format must be one of: mp3, wav, pcm")

        output_format_value: Final = extra_body.get("output_format", "hex")
        if not isinstance(output_format_value, str) or output_format_value not in self._OUTPUT_FORMATS:
            raise ValueError("MiniMax music output format must be one of: url, hex")

        stream_value: Final = extra_body.get("stream", False)
        if not isinstance(stream_value, bool):
            raise ValueError("MiniMax music stream must be a boolean")
        if stream_value and output_format_value != "hex":
            raise ValueError("MiniMax music streaming only supports hex output")

        mapped_params: Final = {
            key: value for key, value in extra_body.items() if key in self._REQUEST_FIELDS and value is not None
        }
        return None, {
            **mapped_params,
            "stream": stream_value,
            "output_format": output_format_value,
            "audio_setting": {**audio_setting, "format": audio_format_value},
        }

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, object],
    ) -> TextToSpeechRequestData:
        prompt_value: Final = optional_params.get("prompt", input)
        request_params: Final = {
            key: value for key, value in optional_params.items() if key in self._REQUEST_FIELDS and key != "prompt"
        }
        prompt_param: Final = {"prompt": prompt_value} if isinstance(prompt_value, str) and prompt_value else {}
        return TextToSpeechRequestData(
            dict_body={"model": model, **prompt_param, **request_params},
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        parsed: Final = self._parse_response(raw_response)
        if parsed.status_code != 0:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=f"MiniMax music generation failed: {parsed.status_message}",
                headers=raw_response.headers,
            )
        if parsed.status != 2:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=f"MiniMax music generation did not complete (status {parsed.status})",
                headers=raw_response.headers,
            )
        audio_content, content_type = self._decode_audio(parsed.audio, raw_response)

        response: Final = httpx.Response(
            status_code=200,
            headers={
                "content-length": str(len(audio_content)),
                "content-type": content_type,
            },
            content=audio_content,
            request=raw_response.request,
        )
        return HttpxBinaryResponseContent(response)

    def _decode_audio(self, audio: str, raw_response: httpx.Response) -> tuple[bytes, str]:
        if audio.startswith(("https://", "http://")):
            return audio.encode(), "text/uri-list"
        try:
            return bytes.fromhex(audio), "application/octet-stream"
        except ValueError as exc:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax music generation returned invalid hex audio",
                headers=raw_response.headers,
            ) from exc

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        from litellm.secret_managers.main import get_secret_str

        base_url: Final = (api_base or get_secret_str("MINIMAX_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        if base_url.endswith(self.ENDPOINT_PATH):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/music_generation"
        return f"{base_url}{self.ENDPOINT_PATH}"

    def _parse_response(self, raw_response: httpx.Response) -> _ParsedMusicResponse:
        try:
            response: Final = _MusicResponse.model_validate(raw_response.json())
            return _ParsedMusicResponse(
                audio=response.data.audio,
                status=response.data.status,
                status_code=response.base_resp.status_code,
                status_message=response.base_resp.status_msg,
            )
        except (json.JSONDecodeError, ValidationError):
            return self._parse_streaming_response(raw_response)

    def _parse_streaming_response(self, raw_response: httpx.Response) -> _ParsedMusicResponse:
        data_lines: Final = tuple(
            line.removeprefix("data:").strip()
            for line in raw_response.text.splitlines()
            if line.startswith("data:") and line.removeprefix("data:").strip()
        )
        try:
            chunks: Final = tuple(_MusicResponse.model_validate_json(line) for line in data_lines)
        except ValidationError as exc:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax music generation returned an invalid response",
                headers=raw_response.headers,
            ) from exc
        if not chunks:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax music generation returned an empty response",
                headers=raw_response.headers,
            )
        final_chunk: Final = chunks[-1]
        return _ParsedMusicResponse(
            audio="".join(chunk.data.audio for chunk in chunks),
            status=final_chunk.data.status,
            status_code=final_chunk.base_resp.status_code,
            status_message=final_chunk.base_resp.status_msg,
        )
