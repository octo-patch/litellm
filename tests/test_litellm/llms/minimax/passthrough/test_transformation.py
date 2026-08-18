"""
Unit tests for MinimaxPassthroughConfig.

Covers the url built for the provider-native voice clone operations, the regional
api base handling, authentication headers and streaming detection.
"""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.minimax.passthrough.transformation import MinimaxPassthroughConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

GLOBAL_API_BASE = "https://api.minimax.io"
REGIONAL_API_BASE = "https://api.minimaxi.com"


class TestMinimaxPassthroughConfig:
    def test_registered_for_the_provider(self):
        config = ProviderConfigManager.get_provider_passthrough_config(
            model="speech-2.8-hd",
            provider=LlmProviders.MINIMAX,
        )

        assert isinstance(config, MinimaxPassthroughConfig)

    @pytest.mark.parametrize(
        "endpoint",
        ["v1/files/upload", "v1/voice_clone", "v1/voice_design"],
    )
    def test_get_complete_url_for_voice_clone_operations(self, monkeypatch, endpoint):
        monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
        config = MinimaxPassthroughConfig()

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="speech-2.8-hd",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url) == f"{GLOBAL_API_BASE}/{endpoint}"
        assert base_target_url == GLOBAL_API_BASE

    @pytest.mark.parametrize(
        "configured_api_base",
        [
            f"{REGIONAL_API_BASE}/v1",
            f"{REGIONAL_API_BASE}/v1/",
            REGIONAL_API_BASE,
        ],
    )
    def test_regional_api_base_does_not_duplicate_the_version(self, monkeypatch, configured_api_base):
        monkeypatch.setenv("MINIMAX_API_BASE", configured_api_base)
        config = MinimaxPassthroughConfig()

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="speech-2.8-hd",
            endpoint="v1/voice_clone",
            request_query_params=None,
            litellm_params={},
        )

        assert str(complete_url) == f"{REGIONAL_API_BASE}/v1/voice_clone"
        assert base_target_url == REGIONAL_API_BASE

    def test_explicit_api_base_wins(self):
        config = MinimaxPassthroughConfig()

        _, base_target_url = config.get_complete_url(
            api_base=f"{REGIONAL_API_BASE}/v1",
            api_key=None,
            model="speech-2.8-hd",
            endpoint="v1/voice_clone",
            request_query_params=None,
            litellm_params={},
        )

        assert base_target_url == REGIONAL_API_BASE

    def test_validate_environment_sets_bearer_token(self):
        config = MinimaxPassthroughConfig()

        headers = config.validate_environment(
            headers={},
            model="speech-2.8-hd",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )

        assert headers == {"Authorization": "Bearer test-key"}

    def test_validate_environment_without_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None, raising=False)
        config = MinimaxPassthroughConfig()

        with pytest.raises(ValueError):
            config.validate_environment(
                headers={},
                model="speech-2.8-hd",
                messages=[],
                optional_params={},
                litellm_params={},
            )

    @pytest.mark.parametrize(
        "request_data, expected",
        [
            ({"file_id": 1, "voice_id": "voice-1", "model": "speech-2.8-hd"}, False),
            ({"stream": False}, False),
            ({"stream": True}, True),
            ({}, False),
        ],
    )
    def test_is_streaming_request(self, request_data, expected):
        config = MinimaxPassthroughConfig()

        assert config.is_streaming_request(endpoint="v1/voice_clone", request_data=request_data) is expected
