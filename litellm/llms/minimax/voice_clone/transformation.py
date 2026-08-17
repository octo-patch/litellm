"""
MiniMax voice clone transformation

Maps the voice clone flow onto the MiniMax voice cloning API:

1. ``POST /v1/files/upload`` with multipart ``file`` + ``purpose`` returns a ``file_id``.
2. ``POST /v1/voice_clone`` with ``file_id``, ``voice_id`` and ``model`` returns the
   registered ``voice_id``.

Reference: https://platform.minimax.io/docs/api-reference/voice-cloning-clone
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
from httpx import Headers

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.voice_clone.transformation import (
    BaseVoiceCloneConfig,
    VoiceCloneFileUploadRequestData,
    VoiceCloneFileUploadResponse,
    VoiceCloneResponse,
)
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import FileTypes
else:
    LiteLLMLoggingObj = Any
    FileTypes = Any


class MinimaxVoiceCloneConfig(BaseVoiceCloneConfig):
    """
    Configuration for MiniMax voice cloning.

    Reference: https://platform.minimax.io/docs/api-reference/voice-cloning-clone

    The reference audio file is uploaded first, then the returned ``file_id`` is
    registered as a cloned voice for one of the supported speech models.
    """

    DEFAULT_API_BASE = "https://api.minimax.io"

    # Regional hosts serving the voice cloning API.
    REGION_API_BASES: dict[str, str] = {
        "global_en": "https://api.minimax.io",
        "cn_zh": "https://api.minimaxi.com",
    }

    FILE_UPLOAD_ENDPOINT_PATH = "/v1/files/upload"
    VOICE_CLONE_ENDPOINT_PATH = "/v1/voice_clone"

    # Upload purposes: the audio to clone, and the optional preview prompt audio.
    CLONE_AUDIO_PURPOSE = "voice_clone"
    PROMPT_AUDIO_PURPOSE = "prompt_audio"
    SUPPORTED_PURPOSES = (CLONE_AUDIO_PURPOSE, PROMPT_AUDIO_PURPOSE)

    SUPPORTED_AUDIO_FORMATS = ("mp3", "m4a", "wav")

    SUPPORTED_VOICE_CLONE_MODELS = (
        "speech-2.8-hd",
        "speech-2.6-hd",
        "speech-02-hd",
        "speech-01-hd",
    )

    def get_supported_voice_clone_params(self, model: str) -> list:
        """
        Fields accepted by the MiniMax voice clone request.
        """
        return ["file_id", "voice_id", "model"]

    def get_supported_voice_clone_models(self) -> list[str]:
        return list(self.SUPPORTED_VOICE_CLONE_MODELS)

    def get_supported_audio_formats(self) -> list[str]:
        return list(self.SUPPORTED_AUDIO_FORMATS)

    def get_api_key_env_var(self) -> str | None:
        return "MINIMAX_API_KEY"

    @classmethod
    def _strip_version_suffix(cls, api_base: str) -> str:
        """
        Normalize a base URL so the endpoint paths are not duplicated.

        ``MINIMAX_API_BASE`` is documented both as a host root and with a trailing
        ``/v1``; the voice cloning endpoint paths already include ``/v1``.
        """
        normalized: Final = api_base.strip().rstrip("/")
        if normalized.endswith("/v1"):
            return normalized[: -len("/v1")]
        return normalized

    @classmethod
    def _resolve_region_api_base(cls, region: str) -> str:
        normalized: Final = region.strip().lower()
        resolved: Final = cls.REGION_API_BASES.get(normalized)
        if resolved is None:
            raise ValueError(
                f"Unsupported MiniMax region '{region}'. Supported regions: {', '.join(sorted(cls.REGION_API_BASES))}."
            )
        return resolved

    @classmethod
    def _resolve_api_base(
        cls,
        api_base: str | None = None,
        litellm_params: dict | None = None,
    ) -> str:
        """
        Resolve the MiniMax host serving the voice cloning API.

        Precedence: explicit ``api_base``, explicit region, ``MINIMAX_API_BASE``,
        ``MINIMAX_API_REGION``, then the default host.
        """
        params: Final = litellm_params or {}

        if isinstance(api_base, str) and api_base.strip():
            return cls._strip_version_suffix(api_base)

        region = params.get("region")
        if isinstance(region, str) and region.strip():
            return cls._strip_version_suffix(cls._resolve_region_api_base(region))

        env_api_base: Final = get_secret_str("MINIMAX_API_BASE")
        if isinstance(env_api_base, str) and env_api_base.strip():
            return cls._strip_version_suffix(env_api_base)

        env_region: Final = get_secret_str("MINIMAX_API_REGION")
        if isinstance(env_region, str) and env_region.strip():
            return cls._strip_version_suffix(cls._resolve_region_api_base(env_region))

        return cls._strip_version_suffix(cls.DEFAULT_API_BASE)

    def get_complete_file_upload_url(
        self,
        api_base: str | None,
        model: str,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        base_url: Final = self._resolve_api_base(api_base=api_base, litellm_params=litellm_params)
        return f"{base_url}{self.FILE_UPLOAD_ENDPOINT_PATH}"

    def get_complete_voice_clone_url(
        self,
        api_base: str | None,
        model: str,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        base_url: Final = self._resolve_api_base(api_base=api_base, litellm_params=litellm_params)
        return f"{base_url}{self.VOICE_CLONE_ENDPOINT_PATH}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Validate the MiniMax environment and set up authentication headers.

        ``Content-Type`` is left to the request transformation: the reference audio
        upload is sent as multipart form data, the clone call as JSON.
        """
        api_key = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")

        if api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})

        return headers

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | Headers,
    ) -> BaseLLMException:
        return MinimaxException(message=error_message, status_code=status_code, headers=headers)

    def _resolve_purpose(self, purpose: str | None) -> str:
        if purpose is None:
            return self.CLONE_AUDIO_PURPOSE

        normalized: Final = purpose.strip()
        if normalized not in self.SUPPORTED_PURPOSES:
            raise ValueError(
                f"Unsupported MiniMax upload purpose '{purpose}'. Supported purposes: "
                f"{', '.join(self.SUPPORTED_PURPOSES)}."
            )
        return normalized

    def _validate_audio_format(self, filename: str) -> None:
        """
        Reject reference audio whose extension is not accepted by MiniMax.

        Files without an extension are passed through; MiniMax validates the content.
        """
        suffix: Final = Path(filename).suffix.lower().lstrip(".")
        if suffix and suffix not in self.SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported reference audio format '{suffix}' for MiniMax voice cloning. "
                f"Supported formats: {', '.join(self.SUPPORTED_AUDIO_FORMATS)}."
            )

    @staticmethod
    def _normalize_model(model: str | None) -> str | None:
        """
        Strip the optional ``minimax/`` prefix without validating the model.
        """
        if not isinstance(model, str) or not model.strip():
            return None

        normalized: Final = model.strip()
        if "/" in normalized:
            return normalized.split("/", 1)[1]
        return normalized

    def _resolve_model(self, model: str | None) -> str:
        normalized: Final = self._normalize_model(model)
        if normalized is None:
            raise ValueError(
                "model is required for MiniMax voice cloning. Supported models: "
                f"{', '.join(self.SUPPORTED_VOICE_CLONE_MODELS)}."
            )

        if normalized not in self.SUPPORTED_VOICE_CLONE_MODELS:
            raise ValueError(
                f"Unsupported model '{normalized}' for MiniMax voice cloning. Supported models: "
                f"{', '.join(self.SUPPORTED_VOICE_CLONE_MODELS)}."
            )
        return normalized

    def transform_voice_clone_file_upload_request(
        self,
        model: str,
        file: FileTypes,
        purpose: str | None = None,
        optional_params: dict | None = None,
        headers: dict | None = None,
        **kwargs,
    ) -> VoiceCloneFileUploadRequestData:
        """
        Build the multipart reference audio upload for MiniMax.

        MiniMax expects ``POST /v1/files/upload`` with:
        - file: the reference audio content
        - purpose: "voice_clone" for the audio to clone, "prompt_audio" for prompt audio
        """
        if file is None:
            raise ValueError("Reference audio file is required for MiniMax voice cloning.")

        resolved_purpose: Final = self._resolve_purpose(purpose)

        extracted: Final = extract_file_data(file)
        filename: Final = extracted["filename"] or f"voice_clone_reference_{int(time.time())}"
        self._validate_audio_format(filename)

        content: Final = extracted["content"]
        content_type: Final = extracted.get("content_type") or "application/octet-stream"

        return VoiceCloneFileUploadRequestData(
            files={
                "file": (filename, content, content_type),
                "purpose": (None, resolved_purpose),
            }
        )

    def _parse_json_response(self, raw_response: httpx.Response) -> dict:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=f"Failed to parse MiniMax response: {e}",
                headers=dict(raw_response.headers),
            )

        if not isinstance(response_json, dict):
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="Unexpected MiniMax response payload, expected a JSON object.",
                headers=dict(raw_response.headers),
            )
        return response_json

    def _raise_for_base_resp(self, response_json: dict, raw_response: httpx.Response) -> None:
        """
        MiniMax reports application level errors in ``base_resp.status_code``,
        where ``0`` means success.
        """
        base_resp: Final = response_json.get("base_resp")
        if not isinstance(base_resp, dict):
            return

        status_code: Final = base_resp.get("status_code")
        if status_code is not None and status_code != 0:
            status_msg: Final = base_resp.get("status_msg") or "Unknown error"
            raise MinimaxException(
                status_code=raw_response.status_code,
                message=f"MiniMax voice clone error: {status_msg} (status_code={status_code})",
                headers=dict(raw_response.headers),
            )

    def transform_voice_clone_file_upload_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None = None,
        **kwargs,
    ) -> VoiceCloneFileUploadResponse:
        """
        Read the ``file_id`` MiniMax assigns to the uploaded reference audio.
        """
        response_json: Final = self._parse_json_response(raw_response)
        self._raise_for_base_resp(response_json, raw_response)

        file_payload: Final = response_json.get("file")
        file_id = file_payload.get("file_id") if isinstance(file_payload, dict) else None
        if file_id is None:
            file_id = response_json.get("file_id")

        if not isinstance(file_id, (int, str)) or (isinstance(file_id, str) and not file_id.strip()):
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax reference audio upload response did not include a file_id.",
                headers=dict(raw_response.headers),
            )

        return VoiceCloneFileUploadResponse(file_id=file_id)

    def transform_voice_clone_request(
        self,
        model: str,
        file_id: int | str,
        voice_id: str,
        optional_params: dict | None = None,
        headers: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Build the MiniMax voice clone request payload.

        MiniMax requires ``file_id`` (from the reference audio upload), the caller
        chosen ``voice_id`` and the ``model`` the cloned voice is registered for.
        """
        # Work on a copy so we don't mutate the caller's dictionary
        params: Final = dict(optional_params) if optional_params else {}
        params.pop("file_id", None)
        params.pop("voice_id", None)
        params.pop("model", None)

        if file_id is None or (isinstance(file_id, str) and not file_id.strip()):
            raise ValueError("file_id is required for MiniMax voice cloning. Upload the reference audio first.")

        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("voice_id is required for MiniMax voice cloning.")

        request_body: Final[dict[str, Any]] = {
            "file_id": file_id.strip() if isinstance(file_id, str) else file_id,
            "voice_id": voice_id.strip(),
            "model": self._resolve_model(model),
        }

        # Handle extra_body for additional MiniMax-specific parameters
        extra_body: Final = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None and key not in request_body:
                    request_body[key] = value

        # Pass through any remaining parameters
        for key, value in params.items():
            if value is not None and key not in request_body:
                request_body[key] = value

        if headers is not None:
            headers.setdefault("Content-Type", "application/json")

        return request_body

    def transform_voice_clone_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None = None,
        voice_id: str | None = None,
        **kwargs,
    ) -> VoiceCloneResponse:
        """
        Transform the MiniMax voice clone response to the standard format.

        MiniMax returns the registered voice together with a status envelope:
        {"voice_id": "...", "base_resp": {"status_code": 0, "status_msg": "success"}}
        """
        response_json: Final = self._parse_json_response(raw_response)
        self._raise_for_base_resp(response_json, raw_response)

        returned_voice_id = response_json.get("voice_id")
        if not isinstance(returned_voice_id, str) or not returned_voice_id.strip():
            # MiniMax registers the caller chosen voice id, so fall back to the
            # requested value when the response omits it.
            returned_voice_id = voice_id.strip() if isinstance(voice_id, str) and voice_id.strip() else None

        if returned_voice_id is None:
            raise MinimaxException(
                status_code=raw_response.status_code,
                message="MiniMax voice clone response did not include a voice_id.",
                headers=dict(raw_response.headers),
            )

        reserved: Final = ("voice_id", "model", "object", "base_resp")
        extra_fields: Final = {key: value for key, value in response_json.items() if key not in reserved}

        response: Final = VoiceCloneResponse(
            voice_id=returned_voice_id.strip(),
            model=self._normalize_model(model),
            **extra_fields,
        )
        response._hidden_params = {"base_resp": response_json.get("base_resp")}
        return response
