import pytest
from unittest.mock import AsyncMock, patch
from reag.client import ReagClient, Document, QueryResult


import json
from litellm.utils import ModelResponse, Choices, Message


@pytest.fixture
def mock_acompletion():
    with patch("reag.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=json.dumps(
                            {
                                "source": {
                                    "content": "Superagent is a workspace for AI-agents.",
                                    "reasoning": "The document describes Superagent.",
                                    "is_irrelevant": False,
                                }
                            }
                        )
                    ),
                )
            ]
        )
        mock_acompletion.return_value = mock_response
        yield mock_acompletion


@pytest.mark.asyncio
async def test_query_with_documents(mock_acompletion):
    async with ReagClient() as client:
        docs = [
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "url": "https://superagent.sh",
                    "source": "web",
                },
            ),
        ]
        response = await client.query("What is Superagent?", documents=docs)
        assert response is not None
        assert len(response) == 1
        result = response[0]
        assert isinstance(result, QueryResult)
        assert result.content is not None
        assert result.document is not None
        assert result.reasoning is not None
        assert isinstance(result.is_irrelevant, bool)


@pytest.mark.asyncio
async def test_query_with_metadata_filter(mock_acompletion):
    async with ReagClient() as client:
        docs = [
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "url": "https://superagent.sh",
                    "source": "web",
                    "id": "sa-1",
                },
            ),
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "url": "https://superagent.sh",
                    "source": "web",
                    "id": "sa-2",
                },
            ),
        ]
        options = {"filter": [{"key": "id", "value": "sa-1", "operator": "equals"}]}
        response = await client.query(
            "What is Superagent?", documents=docs, options=options
        )
        assert response is not None
        assert len(response) == 1
        result = response[0]
        assert isinstance(result, QueryResult)
        assert result.document.metadata["id"] == "sa-1"


@pytest.mark.asyncio
async def test_query_with_integer_filter(mock_acompletion):
    async with ReagClient() as client:
        docs = [
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "version": 1,
                    "source": "web",
                    "id": "sa-1",
                },
            ),
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "version": 2,
                    "source": "web",
                    "id": "sa-2",
                },
            ),
        ]
        options = {
            "filter": [{"key": "version", "value": 2, "operator": "greaterThanOrEqual"}]
        }
        response = await client.query(
            "What is Superagent?", documents=docs, options=options
        )
        assert response is not None
        assert len(response) == 1
        result = response[0]
        assert isinstance(result, QueryResult)
        assert result.document.metadata["version"] == 2


@pytest.fixture
def mock_irrelevant_acompletion():
    with patch("reag.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=json.dumps(
                            {
                                "source": {
                                    "content": "",
                                    "reasoning": "The document is not relevant.",
                                    "is_irrelevant": True,
                                }
                            }
                        )
                    ),
                )
            ]
        )
        mock_acompletion.return_value = mock_response
        yield mock_acompletion


@pytest.mark.asyncio
async def test_query_returns_empty_for_irrelevant_docs(mock_irrelevant_acompletion):
    async with ReagClient() as client:
        docs = [
            Document(
                name="Irrelevant Doc",
                content="This document contains completely unrelated content about cooking recipes.",
                metadata={"type": "recipe", "cuisine": "italian"},
            )
        ]
        response = await client.query("What is Superagent?", documents=docs)
        print(response)
        assert response is not None
        assert len(response) == 0  # Should be empty since doc is irrelevant


@pytest.fixture
def mock_memory_acompletion():
    with patch("reag.client.acompletion", new_callable=AsyncMock) as mock_acompletion:

        async def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            if any("my name is john" in m.get("content", "").lower() for m in messages):
                return ModelResponse(
                    choices=[
                        Choices(
                            finish_reason="stop",
                            index=0,
                            message=Message(
                                content=json.dumps(
                                    {
                                        "source": {
                                            "content": "Your name is John.",
                                            "reasoning": "The user told me their name.",
                                            "is_irrelevant": False,
                                        }
                                    }
                                )
                            ),
                        )
                    ]
                )
            return ModelResponse(
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=Message(
                            content=json.dumps(
                                {
                                    "source": {
                                        "content": "I don't know your name.",
                                        "reasoning": "The user has not told me their name.",
                                        "is_irrelevant": False,
                                    }
                                }
                            )
                        ),
                    )
                ]
            )

        mock_acompletion.side_effect = side_effect
        yield mock_acompletion


@pytest.mark.asyncio
async def test_memory_feature(mock_memory_acompletion):
    from reag.memory import Memory

    memory = Memory()
    async with ReagClient(memory=memory) as client:
        docs = [
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "url": "https://superagent.sh",
                    "source": "web",
                },
            ),
        ]
        await client.query("My name is John", documents=docs)
        response = await client.query("What is my name?", documents=docs)
        assert "john" in response[0].content.lower()


@pytest.mark.asyncio
async def test_streaming_query(mock_acompletion):
    async with ReagClient() as client:
        docs = [
            Document(
                name="Superagent",
                content="Superagent is a workspace for AI-agents that learn, perform work, and collaborate.",
                metadata={
                    "url": "https://superagent.sh",
                    "source": "web",
                },
            ),
        ]
        response_stream = client.query(
            "What is Superagent?", documents=docs, stream=True
        )
        full_response = ""
        async for chunk in await response_stream:
            full_response += chunk.content
        assert "superagent" in full_response.lower()
