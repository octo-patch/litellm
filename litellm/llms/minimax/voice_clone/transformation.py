from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
from litellm.secret_managers.main import get_secret_str
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException


class _MinimaxBaseResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    status_msg: str = ""


class _MinimaxUploadedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: int


class _MinimaxUploadResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    file: _MinimaxUploadedFile
    base_resp: _MinimaxBaseResponse


class _MinimaxVoiceCloneResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    voice_id: str | None = None
    base_resp: _MinimaxBaseResponse


class _MinimaxVoiceCloneRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: int
    voice_id: str
    model: str


@dataclass(frozen=True, slots=True)
class MinimaxVoiceCloneResult:
    file_id: int
    voice_id: str


class MinimaxVoiceCloneConfig:
    DEFAULT_API_BASE = "https://api.minimax.io"
    FILE_UPLOAD_PATH = "/v1/files/upload"
    VOICE_CLONE_PATH = "/v1/voice_clone"
    SUPPORTED_AUDIO_FORMATS = frozenset(("mp3", "m4a", "wav"))
    SUPPORTED_MODELS = frozenset(("speech-2.8-hd", "speech-2.6-hd", "speech-02-hd", "speech-01-hd"))

    def clone_voice(
        self,
        *,
        file_name: str,
        file_content: bytes,
        voice_id: str,
        model: str,
        client: httpx.Client,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> MinimaxVoiceCloneResult:
        self._validate_request(file_name=file_name, voice_id=voice_id, model=model)
        resolved_api_key: Final = self._resolve_api_key(api_key)
        file_id: Final = self._upload_clone_audio(
            client=client,
            file_name=file_name,
            file_content=file_content,
            api_key=resolved_api_key,
            api_base=api_base,
        )
        cloned_voice_id: Final = self._create_voice_clone(
            client=client,
            file_id=file_id,
            voice_id=voice_id,
            model=model,
            api_key=resolved_api_key,
            api_base=api_base,
        )
        return MinimaxVoiceCloneResult(file_id=file_id, voice_id=cloned_voice_id)

    def get_file_upload_url(self, api_base: str | None = None) -> str:
        resolved_api_base: Final = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        return f"{resolved_api_base}{self.FILE_UPLOAD_PATH}"

    def get_voice_clone_url(self, api_base: str | None = None) -> str:
        resolved_api_base: Final = (api_base or self.DEFAULT_API_BASE).rstrip("/")
        return f"{resolved_api_base}{self.VOICE_CLONE_PATH}"

    def _upload_clone_audio(
        self,
        *,
        client: httpx.Client,
        file_name: str,
        file_content: bytes,
        api_key: str,
        api_base: str | None,
    ) -> int:
        response: Final = client.post(
            self.get_file_upload_url(api_base),
            headers=self._authorization_headers(api_key),
            data=MappingProxyType({"purpose": "voice_clone"}),
            files=MappingProxyType({"file": (file_name, file_content)}),
        )
        self._raise_for_http_error(response)
        parsed_response: Final = _MinimaxUploadResponse.model_validate_json(response.content)
        self._raise_for_api_error(parsed_response.base_resp, response)
        return parsed_response.file.file_id

    def _create_voice_clone(
        self,
        *,
        client: httpx.Client,
        file_id: int,
        voice_id: str,
        model: str,
        api_key: str,
        api_base: str | None,
    ) -> str:
        response: Final = client.post(
            self.get_voice_clone_url(api_base),
            headers=MappingProxyType(
                {
                    **self._authorization_headers(api_key),
                    "Content-Type": "application/json",
                }
            ),
            json=_MinimaxVoiceCloneRequest(
                file_id=file_id,
                voice_id=voice_id,
                model=model,
            ).model_dump(),
        )
        self._raise_for_http_error(response)
        parsed_response: Final = _MinimaxVoiceCloneResponse.model_validate_json(response.content)
        self._raise_for_api_error(parsed_response.base_resp, response)
        return parsed_response.voice_id or voice_id

    def _validate_request(self, *, file_name: str, voice_id: str, model: str) -> None:
        file_extension: Final = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if file_extension not in self.SUPPORTED_AUDIO_FORMATS:
            raise ValueError("MiniMax voice cloning requires an mp3, m4a, or wav audio file")
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported MiniMax voice cloning model: {model}")
        if not voice_id.strip():
            raise ValueError("MiniMax voice cloning requires a voice_id")

    def _resolve_api_key(self, api_key: str | None) -> str:
        resolved_api_key: Final = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if not resolved_api_key:
            raise ValueError("MiniMax API key is required")
        return resolved_api_key

    def _authorization_headers(self, api_key: str) -> Mapping[str, str]:
        return MappingProxyType({"Authorization": f"Bearer {api_key}"})

    def _raise_for_http_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        raise MinimaxException(
            status_code=response.status_code,
            message=response.text,
            headers=response.headers,
        )

    def _raise_for_api_error(
        self,
        base_response: _MinimaxBaseResponse,
        response: httpx.Response,
    ) -> None:
        if base_response.status_code == 0:
            return
        raise MinimaxException(
            status_code=response.status_code,
            message=base_response.status_msg,
            headers=response.headers,
        )
