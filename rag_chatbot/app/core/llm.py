import os
import json
import httpx
from typing import Any, Sequence, Generator, AsyncGenerator, Dict
from dotenv import load_dotenv

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ChatResponseGen, CompletionResponse, CompletionResponseGen
from llama_index.core.llms.llm import LLM

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)


class CustomLLM(LLM):
    model: str = "omega"
    max_tokens: int = 4096
    temperature: float = 0.7
    deep_thinking: bool = True

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def _get_headers(self) -> Dict[str, str]:
        api_key = os.getenv("MODELS_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Make sure to set MODELS_API_KEY in your .env file")
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _base_payload(self, messages: Sequence[ChatMessage], stream: bool) -> Dict[str, Any]:
        message_dicts = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        return {
            "model": self.model,
            "messages": message_dicts,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

    def _sync_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(os.getenv('LLM_ENDPOINT'), headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _async_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(os.getenv('LLM_ENDPOINT'), headers=self._get_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        payload = self._base_payload(messages, stream=False)
        data = self._sync_request(payload)
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        return ChatResponse(message=ChatMessage(role="assistant", content=content))

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        payload = self._base_payload(messages, stream=True)

        def gen() -> Generator[ChatResponse, None, None]:
            with httpx.stream(
                "POST",
                os.getenv('LLM_ENDPOINT'),
                headers=self._get_headers(),
                json=payload,
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        piece = json.loads(chunk)
                        choices = piece.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content", "")
                        yield ChatResponse(
                            message=ChatMessage(role="assistant", content=""), delta=delta
                        )
                    except json.JSONDecodeError:
                        continue

        return gen()

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        payload = self._base_payload(messages, stream=False)
        data = await self._async_request(payload)
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        return ChatResponse(message=ChatMessage(role="assistant", content=content))

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        payload = self._base_payload(messages, stream=True)

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            async with httpx.stream(
                "POST",
                os.getenv('LLM_ENDPOINT'),
                headers=self._get_headers(),
                json=payload,
                timeout=60.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        piece = json.loads(chunk)
                        choices = piece.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content", "")
                        yield ChatResponse(
                            message=ChatMessage(role="assistant", content=""), delta=delta
                        )
                    except json.JSONDecodeError:
                        continue

        return gen()

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = self.chat(messages, **kwargs)
        return CompletionResponse(text=chat_response.message.content)

    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        messages = [ChatMessage(role="user", content=prompt)]
        chat_stream = self.stream_chat(messages, **kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            for chat_response in chat_stream:
                yield CompletionResponse(text="", delta=chat_response.delta)

        return gen()

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = await self.achat(messages, **kwargs)
        return CompletionResponse(text=chat_response.message.content)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        messages = [ChatMessage(role="user", content=prompt)]
        chat_stream = self.astream_chat(messages, **kwargs)

        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            async for chat_response in chat_stream:
                yield CompletionResponse(text="", delta=chat_response.delta)

        return gen()
