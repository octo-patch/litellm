"""
Base voice clone transformation configuration.

Voice cloning registers a reusable provider side voice from a reference audio
file. Providers expose this as a two step flow:

1. Upload the reference audio file and receive a provider file handle.
2. Register the cloned voice using that file handle plus a caller chosen voice id.
"""

from typing import TYPE_CHECKING, Any

import httpx
from pydantic import PrivateAttr

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import FileTypes
else:
    LiteLLMLoggingObj = Any
    FileTypes = Any


class VoiceCloneFileUploadRequestData(LiteLLMPydanticObjectBase):
    """Reference audio upload request data."""

    data: dict[str, Any] | None = None
    files: dict[str, Any] | None = None


class VoiceCloneFileUploadResponse(LiteLLMPydanticObjectBase):
    """
    Provider file handle for an uploaded reference audio file.

    ``file_id`` keeps the provider representation (numeric or string) so it can be
    passed back to the voice clone request without lossy coercion.
    """

    file_id: int | str
    object: str = "voice_clone.file"

    model_config = {"extra": "allow"}


class VoiceCloneResponse(LiteLLMPydanticObjectBase):
    """Standard voice clone response format."""

    voice_id: str
    model: str | None = None
    object: str = "voice_clone"

    model_config = {"extra": "allow"}

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class BaseVoiceCloneConfig:
    """
    Base configuration for voice clone transformations.
    Handles provider-agnostic voice clone operations.
    """

    def __init__(self) -> None:
        pass

    def get_supported_voice_clone_params(self, model: str) -> list:
        """
        Get supported voice clone request fields for this provider.
        Override this method in provider-specific implementations.
        """
        return []

    def get_supported_voice_clone_models(self) -> list[str]:
        """
        Get the models a cloned voice can be registered for.
        Override this method in provider-specific implementations.
        """
        return []

    def get_supported_audio_formats(self) -> list[str]:
        """
        Get the reference audio formats accepted by this provider.
        Override this method in provider-specific implementations.
        """
        return []

    def get_api_key_env_var(self) -> str | None:
        """
        Return the provider-specific API key environment variable name, if any.
        """
        return None

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
        Validate environment and return headers.
        Override in provider-specific implementations.
        """
        return headers

    def get_complete_file_upload_url(
        self,
        api_base: str | None,
        model: str,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Get the complete URL used to upload the reference audio file.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("get_complete_file_upload_url must be implemented by provider")

    def get_complete_voice_clone_url(
        self,
        api_base: str | None,
        model: str,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Get the complete URL used to register the cloned voice.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("get_complete_voice_clone_url must be implemented by provider")

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
        Transform the reference audio upload to provider-specific format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_voice_clone_file_upload_request must be implemented by provider")

    def transform_voice_clone_file_upload_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None = None,
        **kwargs,
    ) -> VoiceCloneFileUploadResponse:
        """
        Transform the provider upload response into a provider file handle.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_voice_clone_file_upload_response must be implemented by provider")

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
        Transform the voice clone request to provider-specific format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_voice_clone_request must be implemented by provider")

    def transform_voice_clone_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj | None = None,
        voice_id: str | None = None,
        **kwargs,
    ) -> VoiceCloneResponse:
        """
        Transform the provider voice clone response to standard format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_voice_clone_response must be implemented by provider")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
