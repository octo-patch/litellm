from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    extract_file_data,
)
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import FileTypes


class VoiceCloneUploadRequest(TypedDict):
    data: dict[str, str]
    files: dict[str, tuple[str, bytes, str]]


class VoiceCloneRequest(TypedDict):
    file_id: int
    voice_id: str
    model: str


class MinimaxVoiceCloneConfig:
    API_BASE: Final = "https://api.minimax.io"
    FILE_UPLOAD_PATH: Final = "/v1/files/upload"
    VOICE_CLONE_PATH: Final = "/v1/voice_clone"
    SUPPORTED_AUDIO_FORMATS: Final = frozenset({"mp3", "m4a", "wav"})
    SUPPORTED_MODELS: Final = frozenset({"speech-2.8-hd", "speech-2.6-hd", "speech-02-hd", "speech-01-hd"})

    def validate_environment(self, headers: Mapping[str, str], api_key: str | None = None) -> dict[str, str]:
        resolved_api_key: Final = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if resolved_api_key is None:
            raise ValueError("MiniMax API key is required. Set MINIMAX_API_KEY or pass api_key.")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def get_file_upload_url(self, api_base: str | None = None) -> str:
        resolved_api_base: Final = api_base or self.API_BASE
        return f"{resolved_api_base.rstrip('/')}{self.FILE_UPLOAD_PATH}"

    def get_voice_clone_url(self, api_base: str | None = None) -> str:
        resolved_api_base: Final = api_base or self.API_BASE
        return f"{resolved_api_base.rstrip('/')}{self.VOICE_CLONE_PATH}"

    def transform_file_upload_request(self, file: FileTypes) -> VoiceCloneUploadRequest:
        extracted: Final = extract_file_data(file)
        filename: Final = extracted["filename"]
        if filename is None:
            raise ValueError("MiniMax voice clone audio requires a filename")
        audio_format: Final = Path(filename).suffix.lower().lstrip(".")
        if audio_format not in self.SUPPORTED_AUDIO_FORMATS:
            raise ValueError("MiniMax voice clone audio must be mp3, m4a, or wav")
        return {
            "data": {"purpose": "voice_clone"},
            "files": {
                "file": (
                    filename,
                    extracted["content"],
                    extracted["content_type"],
                )
            },
        }

    def transform_file_upload_response(self, raw_response: httpx.Response) -> int:
        response_data: Final = self._validated_response(raw_response)
        file_data: Final = response_data.get("file")
        if not isinstance(file_data, Mapping):
            raise TypeError("MiniMax file upload response is missing file")
        file_id: Final = file_data.get("file_id")
        if not isinstance(file_id, int):
            raise TypeError("MiniMax file upload response is missing file_id")
        return file_id

    def transform_voice_clone_request(self, file_id: int, voice_id: str, model: str) -> VoiceCloneRequest:
        if file_id <= 0:
            raise ValueError("MiniMax voice clone file_id must be positive")
        if not voice_id:
            raise ValueError("MiniMax voice clone voice_id is required")
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported MiniMax voice clone model: {model}")
        return {"file_id": file_id, "voice_id": voice_id, "model": model}

    def transform_voice_clone_response(self, raw_response: httpx.Response, requested_voice_id: str) -> str:
        response_data: Final = self._validated_response(raw_response)
        response_voice_id: Final = response_data.get("voice_id")
        if isinstance(response_voice_id, str) and response_voice_id:
            return response_voice_id
        if requested_voice_id:
            return requested_voice_id
        raise ValueError("MiniMax voice clone response is missing voice_id")

    @staticmethod
    def _validated_response(raw_response: httpx.Response) -> Mapping[str, object]:
        response_data: Final[object] = raw_response.json()
        if not isinstance(response_data, Mapping):
            raise TypeError("MiniMax voice clone response must be an object")
        base_response: Final = response_data.get("base_resp")
        if not isinstance(base_response, Mapping):
            raise TypeError("MiniMax voice clone response is missing base_resp")
        status_code: Final = base_response.get("status_code")
        if status_code != 0:
            status_message: Final = base_response.get("status_msg")
            message: Final = (
                status_message
                if isinstance(status_message, str) and status_message
                else f"MiniMax voice clone request failed with status {status_code}"
            )
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=message,
                headers=raw_response.headers,
            )
        return response_data
