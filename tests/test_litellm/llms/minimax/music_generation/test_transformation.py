"""
Test MiniMax Music Generation support
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path

import httpx

import litellm
from litellm import music_generation
from litellm.llms.minimax.music_generation.transformation import (
    MinimaxMusicGenerationConfig,
    MinimaxMusicGenerationException,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


def test_minimax_music_generation_config_url():
    """Test that the MiniMax music generation URL is constructed correctly"""
    config = MinimaxMusicGenerationConfig()

    # Default global endpoint
    url = config.get_complete_url(
        model="music-3.0",
        api_base=None,
        litellm_params={},
    )
    assert url == "https://api.minimax.io/v1/music_generation"

    # Custom global api_base
    url = config.get_complete_url(
        model="music-3.0",
        api_base="https://api.minimax.io",
        litellm_params={},
    )
    assert url == "https://api.minimax.io/v1/music_generation"

    # CN region endpoint
    url = config.get_complete_url(
        model="music-3.0",
        api_base="https://api.minimaxi.com/v1",
        litellm_params={},
    )
    assert url == "https://api.minimaxi.com/v1/music_generation"

    # Trailing slash handling
    url = config.get_complete_url(
        model="music-3.0",
        api_base="https://api.minimax.io/",
        litellm_params={},
    )
    assert url == "https://api.minimax.io/v1/music_generation"


def test_minimax_music_generation_supported_params():
    """Test the supported OpenAI params for MiniMax music generation"""
    config = MinimaxMusicGenerationConfig()
    supported = config.get_supported_openai_params("music-3.0")

    assert "lyrics" in supported
    assert "output_format" in supported
    assert "is_instrumental" in supported
    assert "audio_url" in supported
    assert "audio_base64" in supported
    assert "cover_feature_id" in supported


def test_minimax_music_generation_validate_environment():
    """Test that validate_environment sets the Bearer auth header"""
    config = MinimaxMusicGenerationConfig()
    headers = {}
    with patch("litellm.llms.minimax.music_generation.transformation.get_secret_str") as mock_secret:
        mock_secret.return_value = "test-api-key"
        result = config.validate_environment(headers=headers, model="music-3.0")

    assert result["Authorization"] == "Bearer test-api-key"
    assert result["Content-Type"] == "application/json"


def test_minimax_music_generation_validate_environment_no_key():
    """Test that validate_environment raises without an API key"""
    config = MinimaxMusicGenerationConfig()
    with patch("litellm.llms.minimax.music_generation.transformation.get_secret_str") as mock_secret:
        mock_secret.return_value = None
        with pytest.raises(ValueError):
            config.validate_environment(headers={}, model="music-3.0")


def test_minimax_music_generation_request_body():
    """Test that the request body is built with model, prompt and optional fields"""
    config = MinimaxMusicGenerationConfig()

    request_data = config.transform_text_to_speech_request(
        model="music-3.0",
        input="an upbeat lo-fi track",
        voice=None,
        optional_params={
            "lyrics": "la la la",
            "output_format": "url",
            "is_instrumental": False,
            "extra_body": {"stream": False, "lyrics_optimizer": True},
        },
        litellm_params={},
        headers={"Authorization": "Bearer test-key"},
    )

    body = request_data["dict_body"]
    assert body["model"] == "music-3.0"
    assert body["prompt"] == "an upbeat lo-fi track"
    assert body["lyrics"] == "la la la"
    assert body["output_format"] == "url"
    assert body["is_instrumental"] is False
    # extra_body fields are merged through
    assert body["stream"] is False
    assert body["lyrics_optimizer"] is True


def test_minimax_music_generation_response_hex():
    """Test that a hex-encoded audio response is decoded to binary"""
    config = MinimaxMusicGenerationConfig()
    hex_audio = "ffd8ffe0".encode().hex()

    raw = httpx.Response(
        status_code=200,
        json={
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"status": 2, "audio": hex_audio},
        },
        request=httpx.Request("POST", "https://api.minimax.io/v1/music_generation"),
    )

    result = config.transform_text_to_speech_response(
        model="music-3.0", raw_response=raw, logging_obj=MagicMock()
    )
    assert result.content == bytes.fromhex(hex_audio)


def test_minimax_music_generation_response_error():
    """Test that a non-zero base_resp.status_code raises MinimaxMusicGenerationException"""
    config = MinimaxMusicGenerationConfig()

    raw = httpx.Response(
        status_code=200,
        json={
            "base_resp": {"status_code": 1004, "status_msg": "invalid parameter"},
            "data": {"status": 2, "audio": "ffd8"},
        },
        request=httpx.Request("POST", "https://api.minimax.io/v1/music_generation"),
    )

    with pytest.raises(MinimaxMusicGenerationException) as exc_info:
        config.transform_text_to_speech_response(
            model="music-3.0", raw_response=raw, logging_obj=MagicMock()
        )
    assert "invalid parameter" in str(exc_info.value)


def test_minimax_provider_routing():
    """Test that minimax music models are routed to the minimax provider"""
    model, provider, api_key, api_base = get_llm_provider(
        model="minimax/music-3.0", api_base="https://api.minimax.io/v1"
    )
    assert provider == "minimax"
    assert model == "music-3.0"


@patch("litellm.main.base_llm_http_handler.text_to_speech_handler")
def test_minimax_music_generation_dispatch(mock_handler):
    """Test that music_generation dispatches through the base handler"""
    mock_handler.return_value = MagicMock()

    music_generation(
        model="minimax/music-3.0",
        prompt="an upbeat lo-fi track",
        api_key="test-key",
        api_base="https://api.minimax.io/v1",
    )

    mock_handler.assert_called_once()
    kwargs = mock_handler.call_args.kwargs
    assert kwargs["model"] == "music-3.0"
    assert kwargs["input"] == "an upbeat lo-fi track"
    assert kwargs["custom_llm_provider"] == "minimax"
    assert kwargs["_is_async"] is False


if __name__ == "__main__":
    print("Testing MiniMax Music Generation Config...")
    test_minimax_music_generation_config_url()
    print("✓ URL config test passed")

    print("\nTesting MiniMax Music Generation Supported Params...")
    test_minimax_music_generation_supported_params()
    print("✓ Supported params test passed")

    print("\nTesting MiniMax Music Generation Validate Environment...")
    test_minimax_music_generation_validate_environment()
    print("✓ Validate environment test passed")

    print("\nTesting MiniMax Music Generation Request Body...")
    test_minimax_music_generation_request_body()
    print("✓ Request body test passed")

    print("\nTesting MiniMax Music Generation Response (hex)...")
    test_minimax_music_generation_response_hex()
    print("✓ Response hex test passed")

    print("\nTesting MiniMax Music Generation Response (error)...")
    test_minimax_music_generation_response_error()
    print("✓ Response error test passed")

    print("\nTesting MiniMax Music Generation Provider Routing...")
    test_minimax_provider_routing()
    print("✓ Provider routing test passed")

    print("\n✅ All tests passed!")
