from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from workspace_agent.schemas import ModelReply, ToolCall, Usage


_MAX_ATTEMPTS = 3
_MAX_RETRY_DELAY_SECONDS = 1.0


class ModelProtocolError(RuntimeError):
    pass


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must not be blank")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        response: httpx.Response | None = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except httpx.RequestError as error:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"model request failed: {type(error).__name__}"
                    ) from None
                await asyncio.sleep(_backoff_seconds(attempt, None))
                continue

            if response.status_code == 429 or (
                500 <= response.status_code < 600
            ):
                if attempt < _MAX_ATTEMPTS - 1:
                    await asyncio.sleep(
                        _backoff_seconds(
                            attempt,
                            response.headers.get("Retry-After"),
                        )
                    )
                    continue
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"model request failed with HTTP {response.status_code}"
                )
            return _parse_reply(response)

        raise RuntimeError("model request failed after bounded retries")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleModel:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    fallback = 0.05 * (2**attempt)
    parsed = _parse_retry_after(retry_after)
    delay = fallback if parsed is None else parsed
    return min(max(delay, 0.0), _MAX_RETRY_DELAY_SECONDS)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return (retry_at - datetime.now(timezone.utc)).total_seconds()


def _parse_reply(response: httpx.Response) -> ModelReply:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise ModelProtocolError("model response was not valid JSON") from None

    if not isinstance(payload, dict):
        raise ModelProtocolError("model response root must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelProtocolError("model response choices are missing")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ModelProtocolError("model response choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ModelProtocolError("model response message is missing")

    tool_calls = _parse_tool_calls(message)
    usage = _parse_usage(payload)
    return ModelReply(
        message=message,
        tool_calls=tool_calls,
        usage=usage,
    )


def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    if "tool_calls" not in message:
        return ()
    raw_calls = message["tool_calls"]
    if not isinstance(raw_calls, list):
        raise ModelProtocolError("model tool_calls must be an array")

    parsed: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise ModelProtocolError("model tool call must be an object")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise ModelProtocolError("model tool call id is invalid")
        if not isinstance(function, dict):
            raise ModelProtocolError("model tool call function is invalid")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise ModelProtocolError("model tool call name is invalid")
        if not isinstance(raw_arguments, str):
            raise ModelProtocolError("model tool arguments must be JSON text")
        try:
            arguments = json.loads(
                raw_arguments,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            raise ModelProtocolError(
                "model tool arguments were not valid JSON"
            ) from None
        if not isinstance(arguments, dict):
            raise ModelProtocolError("model tool arguments must be an object")
        parsed.append(
            ToolCall(id=call_id, name=name, arguments=arguments)
        )
    return tuple(parsed)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_usage(payload: dict[str, Any]) -> Usage:
    if "usage" not in payload:
        return Usage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
    raw_usage = payload["usage"]
    if not isinstance(raw_usage, dict):
        raise ModelProtocolError("model usage must be an object")

    values: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw_usage.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ModelProtocolError(f"model usage {name} is invalid")
        values[name] = value
    return Usage(**values)
