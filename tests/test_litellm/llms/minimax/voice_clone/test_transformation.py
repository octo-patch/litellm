"""
Test MiniMax voice cloning support
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../"))  # Adds the parent directory to the system path

from litellm.llms.minimax.common_utils import MinimaxException
from litellm.llms.minimax.voice_clone.transformation import MinimaxVoiceCloneConfig

GLOBAL_API_BASE = "https://api.minimax.io/v1"
CHINA_API_BASE = "https://api.minimaxi.com/v1"


def _response(payload: dict, url: str = f"{GLOBAL_API_BASE}/voice_clone", status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", url),
    )


@pytest.mark.parametrize(
    "api_base, expected_upload_url, expected_clone_url",
    [
        (
            GLOBAL_API_BASE,
            f"{GLOBAL_API_BASE}/files/upload",
            f"{GLOBAL_API_BASE}/voice_clone",
        ),
        (
            CHINA_API_BASE,
            f"{CHINA_API_BASE}/files/upload",
            f"{CHINA_API_BASE}/voice_clone",
        ),
        (
            "https://api.minimax.io/",
            f"{GLOBAL_API_BASE}/files/upload",
            f"{GLOBAL_API_BASE}/voice_clone",
        ),
    ],
)
def test_urls_per_region(api_base, expected_upload_url, expected_clone_url):
    config = MinimaxVoiceCloneConfig()

    assert config.get_file_upload_url(api_base) == expected_upload_url
    assert config.get_voice_clone_url(api_base) == expected_clone_url


def test_urls_default_to_global_region(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    config = MinimaxVoiceCloneConfig()

    assert config.get_file_upload_url() == f"{GLOBAL_API_BASE}/files/upload"
    assert config.get_voice_clone_url() == f"{GLOBAL_API_BASE}/voice_clone"


def test_urls_read_api_base_from_environment(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", CHINA_API_BASE)
    config = MinimaxVoiceCloneConfig()

    assert config.get_voice_clone_url() == f"{CHINA_API_BASE}/voice_clone"


def test_validate_environment_uses_bearer_auth():
    assert MinimaxVoiceCloneConfig.validate_environment("my-key") == {"Authorization": "Bearer my-key"}


def test_validate_environment_requires_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("litellm.api_key", None)

    with pytest.raises(ValueError, match="MiniMax API key is required"):
        MinimaxVoiceCloneConfig.validate_environment()


@pytest.mark.parametrize(
    "filename, expected_content_type",
    [("sample.mp3", "audio/mpeg"), ("sample.m4a", "audio/mp4"), ("SAMPLE.WAV", "audio/wav")],
)
def test_file_upload_request(filename, expected_content_type):
    upload = MinimaxVoiceCloneConfig().transform_file_upload_request(
        filename=filename,
        file=b"audio-bytes",
        api_key="my-key",
        api_base=CHINA_API_BASE,
    )

    assert upload.url == f"{CHINA_API_BASE}/files/upload"
    assert upload.headers["Authorization"] == "Bearer my-key"
    assert upload.data == {"purpose": "voice_clone"}
    assert upload.files == {"file": (filename, b"audio-bytes", expected_content_type)}


def test_file_upload_request_supports_prompt_audio_purpose():
    upload = MinimaxVoiceCloneConfig().transform_file_upload_request(
        filename="prompt.wav",
        file=b"audio-bytes",
        purpose="prompt_audio",
        api_key="my-key",
    )

    assert upload.data == {"purpose": "prompt_audio"}


def test_file_upload_request_rejects_unsupported_audio_format():
    with pytest.raises(ValueError, match="mp3, m4a, wav"):
        MinimaxVoiceCloneConfig().transform_file_upload_request(
            filename="sample.ogg",
            file=b"audio-bytes",
            api_key="my-key",
        )


@pytest.mark.parametrize(
    "payload, expected_file_id",
    [
        ({"file": {"file_id": 123456}, "base_resp": {"status_code": 0, "status_msg": "success"}}, 123456),
        ({"file_id": "123456", "base_resp": {"status_code": 0}}, "123456"),
    ],
)
def test_file_upload_response_returns_file_id(payload, expected_file_id):
    file_id = MinimaxVoiceCloneConfig.transform_file_upload_response(_response(payload))

    assert file_id == expected_file_id


def test_file_upload_response_raises_on_base_resp_error():
    payload = {"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}}

    with pytest.raises(MinimaxException, match="1004"):
        MinimaxVoiceCloneConfig.transform_file_upload_response(_response(payload))


def test_file_upload_response_raises_when_file_id_missing():
    payload = {"base_resp": {"status_code": 0}}

    with pytest.raises(MinimaxException, match="did not contain a file_id"):
        MinimaxVoiceCloneConfig.transform_file_upload_response(_response(payload))


def test_voice_clone_request_carries_required_fields():
    request = MinimaxVoiceCloneConfig().transform_voice_clone_request(
        model="speech-2.8-hd",
        file_id=123456,
        voice_id="my_cloned_voice_1",
        api_key="my-key",
        api_base=CHINA_API_BASE,
    )

    assert request.url == f"{CHINA_API_BASE}/voice_clone"
    assert request.headers == {"Authorization": "Bearer my-key", "Content-Type": "application/json"}
    assert request.json_body == {
        "file_id": 123456,
        "voice_id": "my_cloned_voice_1",
        "model": "speech-2.8-hd",
    }


def test_voice_clone_request_strips_provider_prefix_from_model():
    request = MinimaxVoiceCloneConfig().transform_voice_clone_request(
        model="minimax/speech-01-hd",
        file_id=123456,
        voice_id="my_cloned_voice_1",
        api_key="my-key",
    )

    assert request.json_body["model"] == "speech-01-hd"


def test_voice_clone_request_keeps_optional_params_without_dropping_required_fields():
    request = MinimaxVoiceCloneConfig().transform_voice_clone_request(
        model="speech-2.6-hd",
        file_id=123456,
        voice_id="my_cloned_voice_1",
        optional_params={"need_noise_reduction": True, "model": "ignored"},
        api_key="my-key",
    )

    assert request.json_body["need_noise_reduction"] is True
    assert request.json_body["model"] == "speech-2.6-hd"


@pytest.mark.parametrize(
    "model, file_id, voice_id, expected_message",
    [
        ("speech-2.8-turbo", 123456, "my_cloned_voice_1", "MiniMax voice cloning supports"),
        ("speech-2.8-hd", "  ", "my_cloned_voice_1", "requires the file_id"),
        ("speech-2.8-hd", 123456, "   ", "requires a voice_id"),
    ],
)
def test_voice_clone_request_validates_inputs(model, file_id, voice_id, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        MinimaxVoiceCloneConfig().transform_voice_clone_request(
            model=model,
            file_id=file_id,
            voice_id=voice_id,
            api_key="my-key",
        )


def test_voice_clone_response_returns_voice_id():
    payload = {"voice_id": "my_cloned_voice_1", "base_resp": {"status_code": 0, "status_msg": "success"}}

    assert MinimaxVoiceCloneConfig.transform_voice_clone_response(_response(payload)) == "my_cloned_voice_1"


def test_voice_clone_response_raises_on_base_resp_error():
    payload = {"base_resp": {"status_code": 2013, "status_msg": "invalid params"}}

    with pytest.raises(MinimaxException, match="2013"):
        MinimaxVoiceCloneConfig.transform_voice_clone_response(_response(payload))


def test_voice_clone_response_raises_when_voice_id_missing():
    payload = {"base_resp": {"status_code": 0}}

    with pytest.raises(MinimaxException, match="did not contain a voice_id"):
        MinimaxVoiceCloneConfig.transform_voice_clone_response(_response(payload))


def test_voice_clone_response_raises_on_non_json_body():
    raw_response = httpx.Response(
        status_code=502,
        content=b"<html>bad gateway</html>",
        request=httpx.Request("POST", f"{GLOBAL_API_BASE}/voice_clone"),
    )

    with pytest.raises(MinimaxException, match="Failed to parse MiniMax voice cloning response"):
        MinimaxVoiceCloneConfig.transform_voice_clone_response(raw_response)


def test_supported_models_and_audio_formats():
    assert MinimaxVoiceCloneConfig.get_supported_models() == (
        "speech-2.8-hd",
        "speech-2.6-hd",
        "speech-02-hd",
        "speech-01-hd",
    )
    assert MinimaxVoiceCloneConfig.get_supported_audio_formats() == ("mp3", "m4a", "wav")
