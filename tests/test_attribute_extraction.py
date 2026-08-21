import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_pipelines.attribute_extraction import extract_attributes_for_product, ProductAttributes


class FakeResponse:
    def __init__(self, status, json_data=None):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def post(self, url, json, headers):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def make_pbar():
    return MagicMock()


@pytest.mark.asyncio
async def test_successful_extraction_on_first_try():
    claude_payload = {
        "content": [{"text": json.dumps({
            "color": "blue", "material": "cotton", "size": "M",
            "style": "casual", "key_features": ["breathable"],
            "target_audience": "adults", "use_case": "everyday"
        })}]
    }
    session = FakeSession([FakeResponse(200, claude_payload)])
    pbar = make_pbar()

    result = await extract_attributes_for_product(session, "p1", "Blue Shirt", "A comfy shirt", pbar)

    assert isinstance(result, ProductAttributes)
    assert result.product_id == "p1"
    assert result.color == "blue"
    pbar.update.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_returns_empty_attributes_after_all_retries_fail():
    session = FakeSession([
        FakeResponse(500),
        FakeResponse(500),
        FakeResponse(500),
    ])
    pbar = make_pbar()

    result = await extract_attributes_for_product(session, "p2", "Some Product", "Some description", pbar)

    assert isinstance(result, ProductAttributes)
    assert result.product_id == "p2"
    assert result.color is None
    assert result.key_features == []


@pytest.mark.asyncio
async def test_recovers_after_rate_limit_then_success(monkeypatch):
    import llm_pipelines.attribute_extraction as mod
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())

    claude_payload = {
        "content": [{"text": json.dumps({"color": "red"})}]
    }
    session = FakeSession([
        FakeResponse(429),
        FakeResponse(200, claude_payload),
    ])
    pbar = make_pbar()

    result = await extract_attributes_for_product(session, "p3", "Red Item", "desc", pbar)

    assert result.color == "red"
    assert session.call_count == 2


@pytest.mark.asyncio
async def test_product_id_is_always_set_even_on_failure():
    session = FakeSession([FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    pbar = make_pbar()

    result = await extract_attributes_for_product(session, "p4", "title", "desc", pbar)

    assert result.product_id == "p4"
