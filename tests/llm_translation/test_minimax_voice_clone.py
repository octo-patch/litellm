import httpx
import pytest

from litellm.llms.minimax.voice_clone.transformation import (
    MinimaxVoiceCloneConfig,
)


def test_minimax_voice_clone_two_step_transformation() -> None:
    config = MinimaxVoiceCloneConfig()

    assert config.get_file_upload_url() == "https://api.minimax.io/v1/files/upload"
    assert config.get_voice_clone_url("https://api.minimaxi.com") == "https://api.minimaxi.com/v1/voice_clone"

    upload_request = config.transform_file_upload_request(("sample.mp3", b"audio", "audio/mpeg"))
    assert upload_request["data"] == {"purpose": "voice_clone"}
    assert upload_request["files"] == {"file": ("sample.mp3", b"audio", "audio/mpeg")}

    upload_response = httpx.Response(
        200,
        json={
            "file": {"file_id": 123456789},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    file_id = config.transform_file_upload_response(upload_response)
    assert file_id == 123456789

    clone_request = config.transform_voice_clone_request(
        file_id=file_id,
        voice_id="MiniMax001",
        model="speech-2.8-hd",
    )
    assert clone_request == {
        "file_id": 123456789,
        "voice_id": "MiniMax001",
        "model": "speech-2.8-hd",
    }

    clone_response = httpx.Response(
        200,
        json={
            "voice_id": "MiniMax001",
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    assert config.transform_voice_clone_response(clone_response, "MiniMax001") == "MiniMax001"


def test_minimax_voice_clone_rejects_provider_error() -> None:
    config = MinimaxVoiceCloneConfig()
    response = httpx.Response(
        200,
        json={"base_resp": {"status_code": 1004, "status_msg": "invalid file"}},
    )

    with pytest.raises(Exception, match="invalid file"):
        config.transform_file_upload_response(response)
