"""
Test MiniMax voice clone support
"""

import os
import sys
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException
from litellm.llms.minimax.voice_clone.transformation import MinimaxVoiceCloneConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

GET_SECRET_STR_PATH = "litellm.llms.minimax.voice_clone.transformation.get_secret_str"


def _build_response(payload, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.minimax.io/v1/voice_clone"),
    )


class TestMinimaxVoiceCloneRegistration:
    def test_provider_config_manager_returns_voice_clone_config(self):
        config = ProviderConfigManager.get_provider_voice_clone_config(
            model="speech-2.6-hd", provider=LlmProviders.MINIMAX
        )
        assert isinstance(config, MinimaxVoiceCloneConfig)

    def test_provider_config_manager_returns_none_for_other_provider(self):
        config = ProviderConfigManager.get_provider_voice_clone_config(
            model="speech-2.6-hd", provider=LlmProviders.OPENAI
        )
        assert config is None

    def test_supported_metadata(self):
        config = MinimaxVoiceCloneConfig()
        assert config.get_supported_voice_clone_params("speech-2.6-hd") == [
            "file_id",
            "voice_id",
            "model",
        ]
        assert config.get_supported_audio_formats() == ["mp3", "m4a", "wav"]
        assert config.get_supported_voice_clone_models() == [
            "speech-2.8-hd",
            "speech-2.6-hd",
            "speech-02-hd",
            "speech-01-hd",
        ]
        assert config.get_api_key_env_var() == "MINIMAX_API_KEY"


class TestMinimaxVoiceCloneUrls:
    def test_default_urls(self):
        config = MinimaxVoiceCloneConfig()
        with patch(GET_SECRET_STR_PATH, return_value=None):
            assert (
                config.get_complete_file_upload_url(
                    api_base=None, model="speech-2.6-hd"
                )
                == "https://api.minimax.io/v1/files/upload"
            )
            assert (
                config.get_complete_voice_clone_url(
                    api_base=None, model="speech-2.6-hd"
                )
                == "https://api.minimax.io/v1/voice_clone"
            )

    def test_region_selects_host(self):
        config = MinimaxVoiceCloneConfig()
        with patch(GET_SECRET_STR_PATH, return_value=None):
            assert (
                config.get_complete_voice_clone_url(
                    api_base=None,
                    model="speech-2.6-hd",
                    litellm_params={"region": "cn_zh"},
                )
                == "https://api.minimaxi.com/v1/voice_clone"
            )
            assert (
                config.get_complete_voice_clone_url(
                    api_base=None,
                    model="speech-2.6-hd",
                    litellm_params={"region": "global_en"},
                )
                == "https://api.minimax.io/v1/voice_clone"
            )

    def test_unsupported_region_raises(self):
        config = MinimaxVoiceCloneConfig()
        with patch(GET_SECRET_STR_PATH, return_value=None):
            with pytest.raises(ValueError, match="Unsupported MiniMax region"):
                config.get_complete_voice_clone_url(
                    api_base=None,
                    model="speech-2.6-hd",
                    litellm_params={"region": "unknown_region"},
                )

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://api.minimaxi.com",
            "https://api.minimaxi.com/",
            "https://api.minimaxi.com/v1",
            "https://api.minimaxi.com/v1/",
        ],
    )
    def test_api_base_variants_do_not_duplicate_version(self, api_base):
        config = MinimaxVoiceCloneConfig()
        with patch(GET_SECRET_STR_PATH, return_value=None):
            assert (
                config.get_complete_file_upload_url(
                    api_base=api_base, model="speech-2.6-hd"
                )
                == "https://api.minimaxi.com/v1/files/upload"
            )

    def test_env_api_base_is_used(self):
        config = MinimaxVoiceCloneConfig()

        def fake_secret(key: str):
            return "https://api.minimaxi.com/v1" if key == "MINIMAX_API_BASE" else None

        with patch(GET_SECRET_STR_PATH, side_effect=fake_secret):
            assert (
                config.get_complete_voice_clone_url(
                    api_base=None, model="speech-2.6-hd"
                )
                == "https://api.minimaxi.com/v1/voice_clone"
            )

    def test_env_region_is_used(self):
        config = MinimaxVoiceCloneConfig()

        def fake_secret(key: str):
            return "cn_zh" if key == "MINIMAX_API_REGION" else None

        with patch(GET_SECRET_STR_PATH, side_effect=fake_secret):
            assert (
                config.get_complete_voice_clone_url(
                    api_base=None, model="speech-2.6-hd"
                )
                == "https://api.minimaxi.com/v1/voice_clone"
            )

    def test_explicit_api_base_wins_over_region(self):
        config = MinimaxVoiceCloneConfig()
        with patch(GET_SECRET_STR_PATH, return_value=None):
            assert (
                config.get_complete_voice_clone_url(
                    api_base="https://api.minimax.io",
                    model="speech-2.6-hd",
                    litellm_params={"region": "cn_zh"},
                )
                == "https://api.minimax.io/v1/voice_clone"
            )


class TestMinimaxVoiceCloneEnvironment:
    def test_validate_environment_sets_bearer_auth(self):
        config = MinimaxVoiceCloneConfig()
        headers = config.validate_environment(
            headers={}, model="speech-2.6-hd", api_key="test-key"
        )
        assert headers["Authorization"] == "Bearer test-key"
        # Content-Type is set per request: multipart for upload, JSON for clone
        assert "Content-Type" not in headers

    def test_validate_environment_requires_api_key(self):
        config = MinimaxVoiceCloneConfig()
        original_api_key = litellm.api_key
        try:
            litellm.api_key = None
            with patch(GET_SECRET_STR_PATH, return_value=None):
                with pytest.raises(ValueError, match="MiniMax API key is required"):
                    config.validate_environment(headers={}, model="speech-2.6-hd")
        finally:
            litellm.api_key = original_api_key


class TestMinimaxVoiceCloneFileUpload:
    def test_upload_request_defaults_to_voice_clone_purpose(self):
        config = MinimaxVoiceCloneConfig()
        request_data = config.transform_voice_clone_file_upload_request(
            model="speech-2.6-hd",
            file=("sample.mp3", b"audio-bytes", "audio/mpeg"),
        )
        assert request_data.files is not None
        assert request_data.files["file"] == ("sample.mp3", b"audio-bytes", "audio/mpeg")
        assert request_data.files["purpose"] == (None, "voice_clone")
        assert request_data.data is None

    def test_upload_request_supports_prompt_audio_purpose(self):
        config = MinimaxVoiceCloneConfig()
        request_data = config.transform_voice_clone_file_upload_request(
            model="speech-2.6-hd",
            file=("prompt.wav", b"audio-bytes", "audio/wav"),
            purpose="prompt_audio",
        )
        assert request_data.files is not None
        assert request_data.files["purpose"] == (None, "prompt_audio")

    def test_upload_request_rejects_unknown_purpose(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="Unsupported MiniMax upload purpose"):
            config.transform_voice_clone_file_upload_request(
                model="speech-2.6-hd",
                file=("sample.mp3", b"audio-bytes", "audio/mpeg"),
                purpose="assistants",
            )

    @pytest.mark.parametrize("extension", ["mp3", "m4a", "wav"])
    def test_upload_request_accepts_supported_formats(self, extension):
        config = MinimaxVoiceCloneConfig()
        request_data = config.transform_voice_clone_file_upload_request(
            model="speech-2.6-hd",
            file=(f"sample.{extension}", b"audio-bytes", "audio/mpeg"),
        )
        assert request_data.files is not None
        assert request_data.files["file"][0] == f"sample.{extension}"

    def test_upload_request_rejects_unsupported_format(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="Unsupported reference audio format"):
            config.transform_voice_clone_file_upload_request(
                model="speech-2.6-hd",
                file=("sample.ogg", b"audio-bytes", "audio/ogg"),
            )

    def test_upload_request_requires_file(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="Reference audio file is required"):
            config.transform_voice_clone_file_upload_request(
                model="speech-2.6-hd", file=None
            )

    def test_upload_response_reads_nested_file_id(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {
                "file": {"file_id": 123456, "filename": "sample.mp3"},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        upload_response = config.transform_voice_clone_file_upload_response(
            model="speech-2.6-hd", raw_response=raw_response
        )
        # file_id is preserved as returned so it can be sent back unchanged
        assert upload_response.file_id == 123456
        assert upload_response.object == "voice_clone.file"

    def test_upload_response_reads_top_level_file_id(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response({"file_id": "abc-123"})
        upload_response = config.transform_voice_clone_file_upload_response(
            model="speech-2.6-hd", raw_response=raw_response
        )
        assert upload_response.file_id == "abc-123"

    def test_upload_response_without_file_id_raises(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {"base_resp": {"status_code": 0, "status_msg": "success"}}
        )
        with pytest.raises(MinimaxException, match="did not include a file_id"):
            config.transform_voice_clone_file_upload_response(
                model="speech-2.6-hd", raw_response=raw_response
            )

    def test_upload_response_surfaces_base_resp_error(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {
                "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
                "file": {"file_id": 1},
            }
        )
        with pytest.raises(MinimaxException, match="invalid api key"):
            config.transform_voice_clone_file_upload_response(
                model="speech-2.6-hd", raw_response=raw_response
            )


class TestMinimaxVoiceCloneRequest:
    def test_request_contains_required_fields(self):
        config = MinimaxVoiceCloneConfig()
        headers: dict = {}
        request_body = config.transform_voice_clone_request(
            model="speech-2.6-hd",
            file_id=123456,
            voice_id="my-cloned-voice-1",
            headers=headers,
        )
        assert request_body == {
            "file_id": 123456,
            "voice_id": "my-cloned-voice-1",
            "model": "speech-2.6-hd",
        }
        assert headers["Content-Type"] == "application/json"

    def test_request_strips_provider_prefix_from_model(self):
        config = MinimaxVoiceCloneConfig()
        request_body = config.transform_voice_clone_request(
            model="minimax/speech-2.8-hd",
            file_id="123456",
            voice_id="my-cloned-voice-1",
        )
        assert request_body["model"] == "speech-2.8-hd"
        assert request_body["file_id"] == "123456"

    def test_request_requires_file_id(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="file_id is required"):
            config.transform_voice_clone_request(
                model="speech-2.6-hd", file_id=None, voice_id="my-cloned-voice-1"
            )

    def test_request_requires_voice_id(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="voice_id is required"):
            config.transform_voice_clone_request(
                model="speech-2.6-hd", file_id=123456, voice_id="   "
            )

    def test_request_requires_model(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="model is required"):
            config.transform_voice_clone_request(
                model="", file_id=123456, voice_id="my-cloned-voice-1"
            )

    def test_request_rejects_unsupported_model(self):
        config = MinimaxVoiceCloneConfig()
        with pytest.raises(ValueError, match="Unsupported model"):
            config.transform_voice_clone_request(
                model="speech-2.6-turbo",
                file_id=123456,
                voice_id="my-cloned-voice-1",
            )

    def test_request_passes_through_optional_params(self):
        config = MinimaxVoiceCloneConfig()
        request_body = config.transform_voice_clone_request(
            model="speech-2.6-hd",
            file_id=123456,
            voice_id="my-cloned-voice-1",
            optional_params={
                "need_noise_reduction": True,
                "ignored": None,
                "extra_body": {"need_volume_normalization": True},
            },
        )
        assert request_body["need_noise_reduction"] is True
        assert request_body["need_volume_normalization"] is True
        assert "ignored" not in request_body

    def test_request_ignores_conflicting_optional_params(self):
        config = MinimaxVoiceCloneConfig()
        request_body = config.transform_voice_clone_request(
            model="speech-2.6-hd",
            file_id=123456,
            voice_id="my-cloned-voice-1",
            optional_params={
                "file_id": 999,
                "voice_id": "other-voice",
                "model": "speech-01-hd",
            },
        )
        assert request_body["file_id"] == 123456
        assert request_body["voice_id"] == "my-cloned-voice-1"
        assert request_body["model"] == "speech-2.6-hd"


class TestMinimaxVoiceCloneResponse:
    def test_response_parses_voice_id(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {
                "voice_id": "my-cloned-voice-1",
                "input_sensitive": False,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        response = config.transform_voice_clone_response(
            model="speech-2.6-hd", raw_response=raw_response
        )
        assert response.voice_id == "my-cloned-voice-1"
        assert response.model == "speech-2.6-hd"
        assert response.object == "voice_clone"
        assert response.input_sensitive is False

    def test_response_falls_back_to_requested_voice_id(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {"base_resp": {"status_code": 0, "status_msg": "success"}}
        )
        response = config.transform_voice_clone_response(
            model="speech-2.6-hd",
            raw_response=raw_response,
            voice_id="my-cloned-voice-1",
        )
        assert response.voice_id == "my-cloned-voice-1"

    def test_response_without_voice_id_raises(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {"base_resp": {"status_code": 0, "status_msg": "success"}}
        )
        with pytest.raises(MinimaxException, match="did not include a voice_id"):
            config.transform_voice_clone_response(
                model="speech-2.6-hd", raw_response=raw_response
            )

    def test_response_normalizes_prefixed_model_without_validating(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {
                "voice_id": "my-cloned-voice-1",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        response = config.transform_voice_clone_response(
            model="minimax/speech-2.6-hd", raw_response=raw_response
        )
        assert response.model == "speech-2.6-hd"

    def test_response_surfaces_base_resp_error(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = _build_response(
            {
                "voice_id": "my-cloned-voice-1",
                "base_resp": {"status_code": 2013, "status_msg": "invalid params"},
            }
        )
        with pytest.raises(MinimaxException, match="invalid params"):
            config.transform_voice_clone_response(
                model="speech-2.6-hd", raw_response=raw_response
            )

    def test_response_rejects_non_json_payload(self):
        config = MinimaxVoiceCloneConfig()
        raw_response = httpx.Response(
            status_code=200,
            content=b"not json",
            request=httpx.Request("POST", "https://api.minimax.io/v1/voice_clone"),
        )
        with pytest.raises(MinimaxException, match="Failed to parse MiniMax response"):
            config.transform_voice_clone_response(
                model="speech-2.6-hd", raw_response=raw_response
            )

    def test_error_class_is_minimax_exception(self):
        config = MinimaxVoiceCloneConfig()
        error = config.get_error_class(
            error_message="boom", status_code=400, headers={}
        )
        assert isinstance(error, MinimaxException)
