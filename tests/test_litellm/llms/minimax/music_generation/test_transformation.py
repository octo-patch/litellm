from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.minimax.music_generation.transformation import (
    MinimaxMusicGenerationConfig,
)
from litellm.llms.minimax.text_to_speech.transformation import (
    MinimaxException,
    MinimaxTextToSpeechConfig,
)
from litellm.utils import ProviderConfigManager


def _response(payload: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://api.minimax.io/v1/music_generation"),
    )


def test_provider_manager_selects_music_generation_models() -> None:
    for model in MinimaxMusicGenerationConfig.SUPPORTED_MODELS:
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model,
            provider=litellm.LlmProviders.MINIMAX,
        )
        assert isinstance(config, MinimaxMusicGenerationConfig)


def test_provider_manager_preserves_text_to_speech_models() -> None:
    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model="speech-2.6-hd",
        provider=litellm.LlmProviders.MINIMAX,
    )
    assert isinstance(config, MinimaxTextToSpeechConfig)
    assert not isinstance(config, MinimaxMusicGenerationConfig)


def test_speech_dispatches_music_generation_config() -> None:
    with patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.text_to_speech_handler") as handler:
        handler.return_value = MagicMock()
        litellm.speech(  # pyright: ignore[reportUnknownMemberType]  # public API accepts untyped extension parameters
            model="minimax/music-3.0",
            input="Ambient instrumental",
            api_key="test-key",
            response_format="wav",
            extra_body={
                "is_instrumental": True,
                "output_format": "hex",
            },
        )

    call = handler.call_args.kwargs
    assert isinstance(call["text_to_speech_provider_config"], MinimaxMusicGenerationConfig)
    assert call["text_to_speech_optional_params"] == {
        "is_instrumental": True,
        "output_format": "hex",
        "stream": False,
        "audio_setting": {"format": "wav"},
    }


def test_maps_music_generation_request_fields() -> None:
    config = MinimaxMusicGenerationConfig()
    _, optional_params = config.map_openai_params(
        model="music-3.0",
        optional_params={"response_format": "wav"},
        kwargs={
            "extra_body": {
                "lyrics": "[Verse]\nA test song",
                "stream": False,
                "output_format": "url",
                "audio_setting": {"sample_rate": 44100, "bitrate": 256000},
                "lyrics_optimizer": True,
                "is_instrumental": False,
                "aigc_watermark": True,
            }
        },
    )
    request = config.transform_text_to_speech_request(
        model="music-3.0",
        input="Indie folk in a minor key",
        voice=None,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    body = request.get("dict_body")
    assert body == {
        "model": "music-3.0",
        "prompt": "Indie folk in a minor key",
        "lyrics": "[Verse]\nA test song",
        "stream": False,
        "output_format": "url",
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "wav",
        },
        "lyrics_optimizer": True,
        "is_instrumental": False,
        "aigc_watermark": True,
    }


@pytest.mark.parametrize("audio_format", ["mp3", "wav", "pcm"])
def test_supports_documented_audio_formats(audio_format: str) -> None:
    config = MinimaxMusicGenerationConfig()
    _, optional_params = config.map_openai_params(
        model="music-3.0",
        optional_params={"response_format": audio_format},
    )
    assert optional_params["audio_setting"] == {"format": audio_format}


def test_streaming_requires_hex_output() -> None:
    config = MinimaxMusicGenerationConfig()
    with pytest.raises(ValueError, match="streaming only supports hex"):
        config.map_openai_params(
            model="music-3.0",
            optional_params={},
            kwargs={"extra_body": {"stream": True, "output_format": "url"}},
        )


def test_builds_global_and_china_urls() -> None:
    config = MinimaxMusicGenerationConfig()
    assert config.get_complete_url("music-3.0", None, {}) == "https://api.minimax.io/v1/music_generation"
    assert (
        config.get_complete_url("music-3.0", config.CHINA_BASE_URL, {})
        == "https://api.minimaxi.com/v1/music_generation"
    )


def test_parses_hex_audio_response() -> None:
    config = MinimaxMusicGenerationConfig()
    audio = b"generated music"
    result = config.transform_text_to_speech_response(
        model="music-3.0",
        raw_response=_response(
            {
                "data": {"audio": audio.hex(), "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        ),
        logging_obj=MagicMock(),
    )
    assert result.content == audio
    assert result.response.headers["content-type"] == "application/octet-stream"


def test_preserves_url_audio_response() -> None:
    config = MinimaxMusicGenerationConfig()
    audio_url = "https://example.com/generated.mp3"
    result = config.transform_text_to_speech_response(
        model="music-3.0",
        raw_response=_response(
            {
                "data": {"audio": audio_url, "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        ),
        logging_obj=MagicMock(),
    )
    assert result.text == audio_url
    assert result.response.headers["content-type"] == "text/uri-list"


def test_combines_streamed_hex_audio() -> None:
    config = MinimaxMusicGenerationConfig()
    raw_response = httpx.Response(
        200,
        text=(
            'data: {"data":{"audio":"6162","status":1},'
            '"base_resp":{"status_code":0,"status_msg":"success"}}\n\n'
            'data: {"data":{"audio":"6364","status":2},'
            '"base_resp":{"status_code":0,"status_msg":"success"}}\n\n'
        ),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", "https://api.minimax.io/v1/music_generation"),
    )
    result = config.transform_text_to_speech_response(
        model="music-3.0",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )
    assert result.content == b"abcd"


def test_rejects_incomplete_response() -> None:
    config = MinimaxMusicGenerationConfig()
    with pytest.raises(MinimaxException, match="did not complete"):
        config.transform_text_to_speech_response(
            model="music-3.0",
            raw_response=_response(
                {
                    "data": {"audio": "6162", "status": 1},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                }
            ),
            logging_obj=MagicMock(),
        )


def test_surfaces_api_error() -> None:
    config = MinimaxMusicGenerationConfig()
    with pytest.raises(MinimaxException, match="invalid request"):
        config.transform_text_to_speech_response(
            model="music-3.0",
            raw_response=_response(
                {
                    "data": {"audio": "", "status": 2},
                    "base_resp": {
                        "status_code": 1001,
                        "status_msg": "invalid request",
                    },
                }
            ),
            logging_obj=MagicMock(),
        )
