from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    total_tokens: int | None = 0

    def plus(self, other: "Usage") -> "Usage":
        def add(
            left: int | None,
            right: int | None,
        ) -> int | None:
            if left is None or right is None:
                return None
            return left + right

        return Usage(
            prompt_tokens=add(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=add(
                self.completion_tokens,
                other.completion_tokens,
            ),
            total_tokens=add(self.total_tokens, other.total_tokens),
        )

    def as_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    message: dict[str, Any]
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: dict[str, Any]
    summary: str
    error_code: str | None = None

    def model_payload(self) -> dict[str, Any]:
        return {
            "trust_boundary": "UNTRUSTED_WORKSPACE_DATA",
            "ok": self.ok,
            "data": self.data,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    message: str
    model_calls: int
    usage: Usage
