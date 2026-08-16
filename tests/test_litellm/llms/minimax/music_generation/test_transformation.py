"""
Tests for MiniMax music generation (`POST /v1/music_generation`)
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.minimax.music_generation.transformation import (
    MinimaxException,
    MinimaxMusicGenerationConfig,
)
from litellm.llms.minimax.text_to_speech.transformation import MinimaxTextToSpeechConfig
from litellm.utils import ProviderConfigManager

GLOBAL_URL = "https://api.minimax.io/v1/music_generation"
CN_URL = "https://api.minimaxi.com/v1/music_generation"
AUDIO_BYTES = b"\x49\x44\x33\x04music"
AUDIO_HEX = AUDIO_BYTES.hex()


def _raw_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request(method="POST", url=GLOBAL_URL),
    )


class _StubClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requested_urls: list[str] = []

    def get(self, url: str) -> httpx.Response:
        self.requested_urls.append(url)
        return self.response


class TestModelRouting:
    @pytest.mark.parametrize("model", ["music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"])
    def test_music_models_route_to_music_config(self, model):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model, provider=litellm.LlmProviders.MINIMAX
        )
        assert isinstance(config, MinimaxMusicGenerationConfig)

    @pytest.mark.parametrize("model", ["speech-2.8-hd", "speech-02-turbo"])
    def test_speech_models_keep_text_to_speech_config(self, model):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model, provider=litellm.LlmProviders.MINIMAX
        )
        assert isinstance(config, MinimaxTextToSpeechConfig)
        assert not isinstance(config, MinimaxMusicGenerationConfig)


class TestEndpointAndEnvironment:
    @pytest.mark.parametrize(
        "api_base, expected",
        [
            (None, GLOBAL_URL),
            ("https://api.minimax.io", GLOBAL_URL),
            ("https://api.minimax.io/v1", GLOBAL_URL),
            ("https://api.minimax.io/v1/", GLOBAL_URL),
            ("https://api.minimaxi.com", CN_URL),
            ("https://api.minimaxi.com/v1", CN_URL),
        ],
    )
    def test_get_complete_url(self, api_base, expected):
        url = MinimaxMusicGenerationConfig().get_complete_url(model="music-3.0", api_base=api_base, litellm_params={})
        assert url == expected

    def test_validate_environment_sets_bearer_auth(self):
        headers = MinimaxMusicGenerationConfig().validate_environment(
            headers={"x-existing": "kept"}, model="music-3.0", api_key="fake-key"
        )
        assert headers["Authorization"] == "Bearer fake-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["x-existing"] == "kept"

    def test_validate_environment_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            MinimaxMusicGenerationConfig().validate_environment(headers={}, model="music-3.0")


class TestMapOpenAIParams:
    def test_response_format_becomes_audio_format_and_voice_is_dropped(self):
        voice, params = MinimaxMusicGenerationConfig().map_openai_params(
            model="music-3.0",
            optional_params={"response_format": "wav", "speed": 1.5},
            voice="alloy",
        )
        assert voice is None
        assert params["format"] == "wav"
        assert "speed" not in params

    def test_music_fields_are_read_from_kwargs_and_extra_body(self):
        _, params = MinimaxMusicGenerationConfig().map_openai_params(
            model="music-3.0",
            optional_params={},
            kwargs={
                "lyrics": "[Verse]\nline one",
                "is_instrumental": False,
                "litellm_call_id": "ignored",
                "extra_body": {"lyrics_optimizer": True, "unsupported": "ignored"},
            },
        )
        assert params["lyrics"] == "[Verse]\nline one"
        assert params["lyrics_optimizer"] is True
        assert params["is_instrumental"] is False
        assert "litellm_call_id" not in params
        assert "unsupported" not in params

    def test_response_format_overrides_requested_audio_setting_format(self):
        config = MinimaxMusicGenerationConfig()
        _, params = config.map_openai_params(
            model="music-3.0",
            optional_params={"response_format": "pcm"},
            kwargs={"audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}},
        )
        request_data = config.transform_text_to_speech_request(
            model="music-3.0",
            input="ambient pads",
            voice=None,
            optional_params=params,
            litellm_params={},
            headers={},
        )
        assert request_data["dict_body"]["audio_setting"] == {
            "sample_rate": 44100,
            "bitrate": 256000,
            "format": "pcm",
        }
        assert "format" not in request_data["dict_body"]

    def test_unsupported_response_format_raises(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="response_format"):
            MinimaxMusicGenerationConfig().map_openai_params(
                model="music-3.0", optional_params={"response_format": "opus"}
            )

    def test_unsupported_response_format_is_dropped_when_asked(self):
        _, params = MinimaxMusicGenerationConfig().map_openai_params(
            model="music-3.0", optional_params={"response_format": "opus"}, drop_params=True
        )
        assert "format" not in params


class TestTransformRequest:
    def test_prompt_and_hex_default(self):
        request_data = MinimaxMusicGenerationConfig().transform_text_to_speech_request(
            model="music-3.0",
            input="lo-fi piano for a rainy night",
            voice=None,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert request_data["dict_body"] == {
            "model": "music-3.0",
            "prompt": "lo-fi piano for a rainy night",
            "output_format": "hex",
        }

    def test_all_documented_request_fields_are_forwarded(self):
        request_data = MinimaxMusicGenerationConfig().transform_text_to_speech_request(
            model="music-2.6",
            input="upbeat synth pop",
            voice=None,
            optional_params={
                "lyrics": "[Chorus]\nhello",
                "lyrics_optimizer": True,
                "is_instrumental": False,
                "audio_url": "https://example.com/reference.mp3",
                "audio_base64": "AAAA",
                "cover_feature_id": "feature-123",
                "output_format": "url",
                "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "wav"},
            },
            litellm_params={},
            headers={},
        )
        assert request_data["dict_body"] == {
            "model": "music-2.6",
            "prompt": "upbeat synth pop",
            "output_format": "url",
            "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "wav"},
            "lyrics": "[Chorus]\nhello",
            "lyrics_optimizer": True,
            "audio_url": "https://example.com/reference.mp3",
            "audio_base64": "AAAA",
            "cover_feature_id": "feature-123",
            "is_instrumental": False,
        }

    def test_streaming_is_rejected(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="streaming"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_request(
                model="music-3.0",
                input="ambient pads",
                voice=None,
                optional_params={"stream": True},
                litellm_params={},
                headers={},
            )

    def test_unknown_output_format_is_rejected(self):
        with pytest.raises(litellm.UnsupportedParamsError, match="output_format"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_request(
                model="music-3.0",
                input="ambient pads",
                voice=None,
                optional_params={"output_format": "base64"},
                litellm_params={},
                headers={},
            )


class TestTransformResponse:
    def test_hex_audio_is_decoded(self):
        response = MinimaxMusicGenerationConfig().transform_text_to_speech_response(
            model="music-3.0",
            raw_response=_raw_response(
                {"data": {"audio": AUDIO_HEX, "status": 2}, "base_resp": {"status_code": 0, "status_msg": "success"}}
            ),
            logging_obj=None,
        )
        assert response.content == AUDIO_BYTES

    def test_url_audio_is_downloaded(self):
        audio_url = "https://public-cdn.minimax.io/music/track.mp3"
        stub = _StubClient(
            httpx.Response(
                status_code=200,
                content=AUDIO_BYTES,
                headers={"content-type": "audio/mpeg"},
                request=httpx.Request(method="GET", url=audio_url),
            )
        )
        response = MinimaxMusicGenerationConfig(audio_client=stub).transform_text_to_speech_response(
            model="music-3.0",
            raw_response=_raw_response({"data": {"audio": audio_url, "status": 2}, "base_resp": {"status_code": 0}}),
            logging_obj=None,
        )
        assert response.content == AUDIO_BYTES
        assert stub.requested_urls == [audio_url]

    def test_api_error_status_code_raises(self):
        with pytest.raises(MinimaxException, match="status_code 1004"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_response(
                model="music-3.0",
                raw_response=_raw_response({"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}}),
                logging_obj=None,
            )

    def test_incomplete_generation_raises(self):
        with pytest.raises(MinimaxException, match="data.status=1"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_response(
                model="music-3.0",
                raw_response=_raw_response({"data": {"status": 1}, "base_resp": {"status_code": 0}}),
                logging_obj=None,
            )

    def test_missing_audio_raises(self):
        with pytest.raises(MinimaxException, match="no audio payload"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_response(
                model="music-3.0",
                raw_response=_raw_response({"data": {"audio": "", "status": 2}, "base_resp": {"status_code": 0}}),
                logging_obj=None,
            )

    def test_non_hex_audio_raises(self):
        with pytest.raises(MinimaxException, match="failed to decode as hex"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_response(
                model="music-3.0",
                raw_response=_raw_response(
                    {"data": {"audio": "not-hex", "status": 2}, "base_resp": {"status_code": 0}}
                ),
                logging_obj=None,
            )

    def test_non_json_body_raises(self):
        raw_response = httpx.Response(
            status_code=200,
            content=b"<html>gateway error</html>",
            request=httpx.Request(method="POST", url=GLOBAL_URL),
        )
        with pytest.raises(MinimaxException, match="failed to parse as JSON"):
            MinimaxMusicGenerationConfig().transform_text_to_speech_response(
                model="music-3.0", raw_response=raw_response, logging_obj=None
            )
