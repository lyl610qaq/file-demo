from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from workspace_agent.schemas import ModelReply, ToolCall, Usage


_MAX_ATTEMPTS = 3
_MAX_RETRY_DELAY_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)


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

            if response.status_code in _RETRYABLE_STATUS_CODES:
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
        delay = float(value)
    except ValueError:
        pass
    else:
        return delay if math.isfinite(delay) and delay >= 0 else None
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return delay if math.isfinite(delay) and delay >= 0 else None


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

    return validate_model_reply(
        ModelReply(
            message=message,
            tool_calls=None,  # type: ignore[arg-type]
            usage=payload.get("usage"),  # type: ignore[arg-type]
        )
    )


def validate_model_reply(reply: ModelReply) -> ModelReply:
    if not isinstance(reply, ModelReply):
        raise ModelProtocolError("model reply must be a ModelReply")
    if not isinstance(reply.message, dict):
        raise ModelProtocolError("model reply message must be an object")
    if reply.message.get("role") != "assistant":
        raise ModelProtocolError("model reply role must be assistant")

    content = reply.message.get("content")
    if content is not None and not isinstance(content, str):
        raise ModelProtocolError("model reply content is invalid")

    raw_calls_marker = object()
    raw_calls = reply.message.get("tool_calls", raw_calls_marker)
    if raw_calls is None:
        raw_calls = []
    if raw_calls is not raw_calls_marker and not isinstance(raw_calls, list):
        raise ModelProtocolError("model tool_calls must be an array or null")

    parsed_raw_calls: tuple[ToolCall, ...] | None = None
    if isinstance(raw_calls, list):
        parsed_raw_calls = tuple(
            _validate_wire_tool_call(raw_call) for raw_call in raw_calls
        )
    provided_calls = _validate_provided_tool_calls(reply.tool_calls)
    if parsed_raw_calls is not None:
        if provided_calls is not None and provided_calls != parsed_raw_calls:
            raise ModelProtocolError("model tool call representations differ")
        tool_calls = parsed_raw_calls
    else:
        tool_calls = provided_calls or ()

    seen_ids: set[str] = set()
    canonical_raw_calls: list[dict[str, Any]] = []
    canonical_calls: list[ToolCall] = []
    for call in tool_calls:
        if call.id in seen_ids:
            raise ModelProtocolError("model tool call ids must be unique")
        seen_ids.add(call.id)
        arguments, encoded_arguments = _canonical_arguments(call.arguments)
        canonical_calls.append(
            ToolCall(
                id=call.id,
                name=call.name,
                arguments=arguments,
            )
        )
        canonical_raw_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": encoded_arguments,
                },
            }
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if canonical_raw_calls:
        message["tool_calls"] = canonical_raw_calls
    return ModelReply(
        message=message,
        tool_calls=tuple(canonical_calls),
        usage=_canonical_usage(reply.usage),
    )


def _validate_wire_tool_call(raw_call: Any) -> ToolCall:
    if not isinstance(raw_call, dict):
        raise ModelProtocolError("model tool call must be an object")
    if raw_call.get("type") != "function":
        raise ModelProtocolError("model tool call type must be function")
    call_id = raw_call.get("id")
    function = raw_call.get("function")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ModelProtocolError("model tool call id is invalid")
    if not isinstance(function, dict):
        raise ModelProtocolError("model tool call function is invalid")
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or not name.strip():
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
    canonical_arguments, _ = _canonical_arguments(arguments)
    return ToolCall(call_id, name, canonical_arguments)


def _validate_provided_tool_calls(
    raw_calls: tuple[ToolCall, ...] | None,
) -> tuple[ToolCall, ...] | None:
    if raw_calls is None:
        return None
    if not isinstance(raw_calls, (tuple, list)):
        raise ModelProtocolError("model tool_calls must be a sequence or null")
    validated: list[ToolCall] = []
    for call in raw_calls:
        if not isinstance(call, ToolCall):
            raise ModelProtocolError("model tool call is invalid")
        if not isinstance(call.id, str) or not call.id.strip():
            raise ModelProtocolError("model tool call id is invalid")
        if not isinstance(call.name, str) or not call.name.strip():
            raise ModelProtocolError("model tool call name is invalid")
        arguments, _ = _canonical_arguments(call.arguments)
        validated.append(ToolCall(call.id, call.name, arguments))
    return tuple(validated)


def _canonical_arguments(
    arguments: Any,
) -> tuple[dict[str, Any], str]:
    if not isinstance(arguments, dict):
        raise ModelProtocolError("model tool arguments must be an object")
    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        canonical = json.loads(
            encoded,
            parse_constant=_reject_json_constant,
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        json.JSONDecodeError,
    ):
        raise ModelProtocolError(
            "model tool arguments were not strict JSON"
        ) from None
    if not isinstance(canonical, dict):
        raise ModelProtocolError("model tool arguments must be an object")
    return canonical, encoded


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _canonical_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
    if isinstance(raw_usage, Usage):
        source = raw_usage.as_dict()
    elif isinstance(raw_usage, dict):
        source = raw_usage
    else:
        raise ModelProtocolError("model usage must be an object")

    values: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = source.get(name)
        if value is not None and (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ModelProtocolError(f"model usage {name} is invalid")
        values[name] = value
    return Usage(**values)
