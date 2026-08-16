"""
Test MiniMax pass-through support for the voice cloning endpoints
"""

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.minimax.passthrough.transformation import MinimaxPassthroughConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

VOICE_CLONE_ENDPOINT = "v1/voice_clone"
FILE_UPLOAD_ENDPOINT = "v1/files/upload"
VOICE_CLONE_MODEL = "speech-2.8-hd"


@pytest.fixture(autouse=True)
def clean_minimax_env(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)


def _get_url(api_base, endpoint, request_query_params=None):
    url, base_target_url = MinimaxPassthroughConfig().get_complete_url(
        api_base=api_base,
        api_key="fake-key",
        model=VOICE_CLONE_MODEL,
        endpoint=endpoint,
        request_query_params=request_query_params,
        litellm_params={},
    )
    return str(url), base_target_url


def test_minimax_passthrough_config_is_registered():
    """The provider registry must resolve MiniMax pass-through requests to the MiniMax config"""
    config = ProviderConfigManager.get_provider_passthrough_config(
        model=VOICE_CLONE_MODEL,
        provider=LlmProviders.MINIMAX,
    )

    assert isinstance(config, MinimaxPassthroughConfig)


def test_voice_clone_url_defaults_to_global_endpoint():
    url, base_target_url = _get_url(api_base=None, endpoint=VOICE_CLONE_ENDPOINT)

    assert url == "https://api.minimax.io/v1/voice_clone"
    assert base_target_url == "https://api.minimax.io"


def test_file_upload_url_keeps_leading_slash_endpoints_intact():
    url, _ = _get_url(api_base=None, endpoint=f"/{FILE_UPLOAD_ENDPOINT}")

    assert url == "https://api.minimax.io/v1/files/upload"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.minimaxi.com",
        "https://api.minimaxi.com/",
        "https://api.minimaxi.com/v1",
    ],
)
def test_voice_clone_url_uses_regional_api_base_without_duplicating_version(api_base):
    """A regional api base must not produce a duplicated /v1 segment"""
    url, _ = _get_url(api_base=api_base, endpoint=VOICE_CLONE_ENDPOINT)

    assert url == "https://api.minimaxi.com/v1/voice_clone"


def test_voice_clone_url_reads_regional_api_base_from_environment(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", "https://api.minimaxi.com/v1")

    url, _ = _get_url(api_base=None, endpoint=VOICE_CLONE_ENDPOINT)

    assert url == "https://api.minimaxi.com/v1/voice_clone"


def test_voice_clone_url_preserves_query_params():
    url, _ = _get_url(
        api_base=None,
        endpoint=VOICE_CLONE_ENDPOINT,
        request_query_params={"GroupId": "1234"},
    )

    assert url == "https://api.minimax.io/v1/voice_clone?GroupId=1234"


def test_validate_environment_sends_bearer_token_and_keeps_existing_headers(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    headers = MinimaxPassthroughConfig().validate_environment(
        headers={"x-request-id": "abc"},
        model=VOICE_CLONE_MODEL,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert headers == {"x-request-id": "abc", "Authorization": "Bearer env-key"}


def test_validate_environment_does_not_pin_content_type():
    """The audio upload is multipart, so the pass-through must not force a json content type"""
    headers = MinimaxPassthroughConfig().validate_environment(
        headers={},
        model=VOICE_CLONE_MODEL,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="fake-key",
        api_base=None,
    )

    assert "Content-Type" not in headers


def test_validate_environment_raises_without_api_key():
    with pytest.raises(ValueError, match="MiniMax API key is required"):
        MinimaxPassthroughConfig().validate_environment(
            headers={},
            model=VOICE_CLONE_MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )


def test_voice_clone_request_is_not_streaming():
    config = MinimaxPassthroughConfig()

    assert (
        config.is_streaming_request(
            endpoint=VOICE_CLONE_ENDPOINT,
            request_data={
                "file_id": 12345,
                "voice_id": "customVoice001",
                "model": VOICE_CLONE_MODEL,
            },
        )
        is False
    )
    assert config.is_streaming_request(endpoint=VOICE_CLONE_ENDPOINT, request_data={"stream": True}) is True


def test_voice_clone_request_is_routed_to_minimax():
    """A voice clone pass-through call must reach the MiniMax voice clone endpoint and return its voice_id"""
    sent_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "voice_id": "customVoice001",
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    response = litellm.llm_passthrough_route(
        method="POST",
        endpoint=VOICE_CLONE_ENDPOINT,
        model=f"minimax/{VOICE_CLONE_MODEL}",
        api_key="fake-key",
        json={
            "file_id": 12345,
            "voice_id": "customVoice001",
            "model": VOICE_CLONE_MODEL,
        },
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler))),
    )

    assert len(sent_requests) == 1
    assert sent_requests[0].method == "POST"
    assert str(sent_requests[0].url) == "https://api.minimax.io/v1/voice_clone"
    assert sent_requests[0].headers["Authorization"] == "Bearer fake-key"
    assert response.json()["voice_id"] == "customVoice001"
