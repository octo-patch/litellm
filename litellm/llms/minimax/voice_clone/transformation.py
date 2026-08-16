"""
MiniMax voice cloning transformation

Uploads reference audio to MiniMax's file endpoint, then registers the cloned voice.

Reference: https://platform.minimax.io/docs/api-reference/voice-cloning-clone
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import Self

import litellm
from litellm.llms.minimax.common_utils import MinimaxException, resolve_minimax_api_host
from litellm.secret_managers.main import get_secret_str

MinimaxVoiceCloneFilePurpose: TypeAlias = Literal["voice_clone", "prompt_audio"]

_API_VERSION_SUFFIX: Final = "/v1"
_FILE_UPLOAD_PATH: Final = "/files/upload"
_VOICE_CLONE_PATH: Final = "/voice_clone"
_MODEL_PREFIX: Final = "minimax/"
_SUCCESS_STATUS_CODE: Final = 0
_SUPPORTED_MODELS: Final = ("speech-2.8-hd", "speech-2.6-hd", "speech-02-hd", "speech-01-hd")
_AUDIO_CONTENT_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {"mp3": "audio/mpeg", "m4a": "audio/mp4", "wav": "audio/wav"}
)
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class MinimaxVoiceCloneFileUpload:
    """A multipart upload of the reference audio, ready to hand to an http client."""

    url: str
    headers: Mapping[str, str]
    data: Mapping[str, str]
    files: Mapping[str, tuple[str, bytes, str]]


@dataclass(frozen=True, slots=True)
class MinimaxVoiceCloneRequest:
    """A /v1/voice_clone call, ready to hand to an http client."""

    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, object]


class _BaseResp(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status_code: int | None = None
    status_msg: str | None = None


class _MinimaxResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    base_resp: _BaseResp | None = None

    @classmethod
    def parse(cls, raw_response: httpx.Response) -> Self:
        try:
            parsed: Final = cls.model_validate_json(raw_response.text)
        except ValidationError as parse_error:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=f"Failed to parse MiniMax voice cloning response: {parse_error}",
                headers=raw_response.headers,
            ) from parse_error
        parsed.raise_for_status(raw_response)
        return parsed

    def raise_for_status(self, raw_response: httpx.Response) -> None:
        """MiniMax reports failures in ``base_resp`` while still answering with HTTP 200."""
        status_code: Final = self.base_resp.status_code if self.base_resp is not None else None
        if status_code is None or status_code == _SUCCESS_STATUS_CODE:
            return
        status_msg: Final = self.base_resp.status_msg if self.base_resp is not None else None
        raise MinimaxException(
            status_code=raw_response.status_code,
            message=f"MiniMax voice cloning error {status_code}: {status_msg or 'unknown error'}",
            headers=raw_response.headers,
        )


class _UploadedFile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    file_id: int | str | None = None


class _FileUploadResponse(_MinimaxResponse):
    file: _UploadedFile | None = None
    file_id: int | str | None = None


class _VoiceCloneResponse(_MinimaxResponse):
    voice_id: str | None = None


class MinimaxVoiceCloneConfig:
    """
    Configuration for MiniMax voice cloning

    Reference audio goes to /v1/files/upload with a ``voice_clone`` or ``prompt_audio`` purpose,
    then the returned ``file_id`` is registered against a caller chosen ``voice_id`` on /v1/voice_clone

    Set ``api_base`` (or MINIMAX_API_BASE) to https://api.minimaxi.com/v1 for the China region
    """

    @staticmethod
    def get_supported_models() -> tuple[str, ...]:
        return _SUPPORTED_MODELS

    @staticmethod
    def get_supported_audio_formats() -> tuple[str, ...]:
        return tuple(_AUDIO_CONTENT_TYPES)

    @staticmethod
    def _resolve_api_base(api_base: str | None) -> str:
        return f"{resolve_minimax_api_host(api_base)}{_API_VERSION_SUFFIX}"

    @staticmethod
    def validate_environment(api_key: str | None = None) -> Mapping[str, str]:
        resolved_api_key: Final = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )
        return MappingProxyType({"Authorization": f"Bearer {resolved_api_key}"})

    def get_file_upload_url(self, api_base: str | None = None) -> str:
        return f"{self._resolve_api_base(api_base)}{_FILE_UPLOAD_PATH}"

    def get_voice_clone_url(self, api_base: str | None = None) -> str:
        return f"{self._resolve_api_base(api_base)}{_VOICE_CLONE_PATH}"

    def transform_file_upload_request(
        self,
        filename: str,
        file: bytes,
        purpose: MinimaxVoiceCloneFilePurpose = "voice_clone",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> MinimaxVoiceCloneFileUpload:
        extension: Final = filename.rsplit(".", 1)[-1].lower()
        content_type: Final = _AUDIO_CONTENT_TYPES.get(extension)
        if content_type is None:
            raise ValueError(
                f"MiniMax voice cloning accepts {', '.join(_AUDIO_CONTENT_TYPES)} reference audio, got {filename!r}."
            )
        return MinimaxVoiceCloneFileUpload(
            url=self.get_file_upload_url(api_base),
            headers=self.validate_environment(api_key),
            data=MappingProxyType({"purpose": purpose}),
            files=MappingProxyType({"file": (filename, file, content_type)}),
        )

    @staticmethod
    def transform_file_upload_response(raw_response: httpx.Response) -> int | str:
        parsed: Final = _FileUploadResponse.parse(raw_response)
        nested_file_id: Final = parsed.file.file_id if parsed.file is not None else None
        file_id: Final = parsed.file_id if nested_file_id is None else nested_file_id
        if file_id is None or file_id == "":
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax reference audio upload response did not contain a file_id.",
                headers=raw_response.headers,
            )
        return file_id

    def transform_voice_clone_request(
        self,
        model: str,
        file_id: int | str,
        voice_id: str,
        optional_params: Mapping[str, object] = _NO_PARAMS,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> MinimaxVoiceCloneRequest:
        resolved_model: Final = model.removeprefix(_MODEL_PREFIX)
        if resolved_model not in _SUPPORTED_MODELS:
            raise ValueError(f"MiniMax voice cloning supports {', '.join(_SUPPORTED_MODELS)}, got {model!r}.")
        if not str(file_id).strip():
            raise ValueError("MiniMax voice cloning requires the file_id returned by the reference audio upload.")
        if not voice_id.strip():
            raise ValueError("MiniMax voice cloning requires a voice_id to register the cloned voice under.")
        return MinimaxVoiceCloneRequest(
            url=self.get_voice_clone_url(api_base),
            headers=MappingProxyType({**self.validate_environment(api_key), "Content-Type": "application/json"}),
            json_body=MappingProxyType(
                {**optional_params, "file_id": file_id, "voice_id": voice_id, "model": resolved_model}
            ),
        )

    @staticmethod
    def transform_voice_clone_response(raw_response: httpx.Response) -> str:
        parsed: Final = _VoiceCloneResponse.parse(raw_response)
        if not parsed.voice_id:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax voice cloning response did not contain a voice_id.",
                headers=raw_response.headers,
            )
        return parsed.voice_id
