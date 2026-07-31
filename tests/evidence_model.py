from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any

from workspace_agent.loop import SYSTEM_POLICY
from workspace_agent.schemas import ModelReply, ToolCall


class EvidenceError(RuntimeError):
    pass


class EvidenceDrivenModel:
    def __init__(self, task: str) -> None:
        if task not in {"falcon", "archive", "inspect"}:
            raise ValueError("unknown evidence task")
        self.task = task
        self.system_policy = ""
        self.read_paths: set[str] = set()
        self.obsolete_paths: set[str] = set()
        self.successful_moves: set[str] = set()
        self.generated_content = ""
        self.observed_injection = False
        self._counter = 0
        self._expected: ToolCall | None = None
        self._phase = "initial"
        self._pending_paths: list[str] = []
        self._documents: dict[str, str] = {}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        self._validate_request(messages, tools)
        if self._phase == "initial":
            return self._start()

        call, data = self._consume_tool_evidence(messages)
        if self.task == "falcon":
            return self._continue_falcon(call, data)
        if self.task == "archive":
            return self._continue_archive(call, data)
        return self._continue_inspection(call, data)

    def _validate_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> None:
        if len(messages) < 2 or messages[0] != {
            "role": "system",
            "content": SYSTEM_POLICY,
        }:
            raise EvidenceError("system policy evidence is missing")
        if not isinstance(tools, list) or not tools:
            raise EvidenceError("tool schema evidence is missing")
        self.system_policy = str(messages[0]["content"])

    def _start(self) -> ModelReply:
        if self.task == "falcon":
            self._phase = "falcon-search"
            return self._call(
                "search_files",
                {
                    "query": "Project Falcon",
                    "path": ".",
                    "limit": 100,
                    "case_sensitive": True,
                },
            )
        if self.task == "archive":
            self._phase = "archive-list"
            return self._call(
                "list_dir",
                {
                    "path": "drafts",
                    "recursive": True,
                    "limit": 100,
                },
            )
        self._phase = "inspect-search"
        return self._call(
            "search_files",
            {
                "query": "IGNORE ALL PRIOR INSTRUCTIONS",
                "path": "adversarial",
                "limit": 10,
                "case_sensitive": True,
            },
        )

    def _consume_tool_evidence(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[ToolCall, dict[str, Any]]:
        expected = self._expected
        if expected is None or len(messages) < 2:
            raise EvidenceError("tool evidence is missing")
        assistant = messages[-2]
        tool_message = messages[-1]
        if (
            assistant.get("role") != "assistant"
            or tool_message.get("role") != "tool"
            or tool_message.get("tool_call_id") != expected.id
        ):
            raise EvidenceError("tool evidence is missing or mismatched")
        calls = assistant.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise EvidenceError("tool evidence call is invalid")
        raw_call = calls[0]
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if (
            not isinstance(function, dict)
            or raw_call.get("id") != expected.id
            or raw_call.get("type") != "function"
            or function.get("name") != expected.name
        ):
            raise EvidenceError("tool evidence call is invalid")
        try:
            assistant_arguments = json.loads(function["arguments"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise EvidenceError("tool evidence arguments are invalid") from None
        if assistant_arguments != expected.arguments:
            raise EvidenceError("tool evidence arguments are mismatched")

        content = tool_message.get("content")
        if not isinstance(content, str):
            raise EvidenceError("tool evidence payload is missing")
        try:
            payload = json.loads(content, parse_constant=self._reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise EvidenceError("tool evidence payload is invalid") from None
        if (
            not isinstance(payload, dict)
            or payload.get("trust_boundary") != "UNTRUSTED_WORKSPACE_DATA"
            or payload.get("ok") is not True
            or not isinstance(payload.get("data"), dict)
            or payload.get("error_code") is not None
        ):
            raise EvidenceError("tool evidence payload is untrusted or failed")
        self._expected = None
        return expected, payload["data"]

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    def _continue_falcon(
        self,
        call: ToolCall,
        data: dict[str, Any],
    ) -> ModelReply:
        if self._phase == "falcon-search" and call.name == "search_files":
            self._require_complete_page(data)
            matches = data.get("matches")
            if not isinstance(matches, list):
                raise EvidenceError("tool evidence search matches are invalid")
            paths = {
                match.get("path")
                for match in matches
                if isinstance(match, dict)
                and isinstance(match.get("path"), str)
                and match["path"].endswith(".md")
            }
            if not paths or None in paths:
                raise EvidenceError("tool evidence search returned no files")
            self._pending_paths = sorted(paths)
            self._phase = "falcon-read"
            return self._read_next()

        if self._phase == "falcon-read" and call.name == "read_file":
            path, content = self._read_evidence(call, data)
            self._documents[path] = content
            if self._pending_paths:
                return self._read_next()
            self.generated_content = self._build_falcon_index()
            self._phase = "falcon-write"
            return self._call(
                "write_file",
                {
                    "path": "falcon_index.md",
                    "content": self.generated_content,
                    "overwrite": False,
                },
            )

        if self._phase == "falcon-write" and call.name == "write_file":
            self._validate_write(data, "falcon_index.md", self.generated_content)
            self._phase = "done"
            return self._final("Falcon/Aquila index created from tool evidence.")
        raise EvidenceError("tool evidence arrived in an invalid Falcon phase")

    def _continue_archive(
        self,
        call: ToolCall,
        data: dict[str, Any],
    ) -> ModelReply:
        if self._phase == "archive-list" and call.name == "list_dir":
            self._require_complete_page(data)
            entries = data.get("entries")
            if not isinstance(entries, list):
                raise EvidenceError("tool evidence directory entries are invalid")
            paths = {
                entry.get("path")
                for entry in entries
                if isinstance(entry, dict)
                and entry.get("type") == "file"
                and isinstance(entry.get("path"), str)
                and entry["path"].startswith("drafts/")
                and entry["path"].endswith(".md")
            }
            if not paths or None in paths:
                raise EvidenceError("tool evidence returned no draft files")
            self._pending_paths = sorted(paths)
            self._phase = "archive-read"
            return self._read_next()

        if self._phase == "archive-read" and call.name == "read_file":
            path, content = self._read_evidence(call, data)
            status = self._frontmatter_status(content)
            if status == "obsolete":
                self.obsolete_paths.add(path)
            if self._pending_paths:
                return self._read_next()
            self._pending_paths = sorted(self.obsolete_paths)
            self._phase = "archive-move"
            return self._move_next_or_manifest()

        if self._phase == "archive-move" and call.name == "move_file":
            source = data.get("source")
            destination = data.get("destination")
            if (
                source != call.arguments.get("source")
                or destination != call.arguments.get("destination")
                or not isinstance(destination, str)
            ):
                raise EvidenceError("tool evidence move result is mismatched")
            self.successful_moves.add(destination)
            return self._move_next_or_manifest()

        if self._phase == "archive-write" and call.name == "write_file":
            self._validate_write(data, "archive/MANIFEST.md", self.generated_content)
            self._phase = "done"
            return self._final("Obsolete drafts archived from tool evidence.")
        raise EvidenceError("tool evidence arrived in an invalid archive phase")

    def _continue_inspection(
        self,
        call: ToolCall,
        data: dict[str, Any],
    ) -> ModelReply:
        if self._phase == "inspect-search" and call.name == "search_files":
            self._require_complete_page(data)
            matches = data.get("matches")
            if not isinstance(matches, list):
                raise EvidenceError("tool evidence search matches are invalid")
            paths = {
                match.get("path")
                for match in matches
                if isinstance(match, dict)
                and isinstance(match.get("path"), str)
            }
            if not paths or None in paths:
                raise EvidenceError("tool evidence search returned no files")
            self._pending_paths = sorted(paths)
            self._phase = "inspect-read"
            return self._read_next()

        if self._phase == "inspect-read" and call.name == "read_file":
            _, content = self._read_evidence(call, data)
            if "IGNORE ALL PRIOR INSTRUCTIONS" in content:
                self.observed_injection = True
            if self._pending_paths:
                return self._read_next()
            if not self.observed_injection:
                raise EvidenceError("tool evidence did not contain the fixture")
            self._phase = "done"
            return self._final("The injection text was treated only as data.")
        raise EvidenceError("tool evidence arrived in an invalid inspection phase")

    def _read_next(self) -> ModelReply:
        if not self._pending_paths:
            raise EvidenceError("tool evidence read queue is empty")
        path = self._pending_paths.pop(0)
        return self._call(
            "read_file",
            {"path": path, "offset": 0, "limit": 16_384},
        )

    def _read_evidence(
        self,
        call: ToolCall,
        data: dict[str, Any],
    ) -> tuple[str, str]:
        self._require_complete_page(data)
        path = data.get("path")
        content = data.get("content")
        if path != call.arguments.get("path") or not isinstance(content, str):
            raise EvidenceError("tool evidence read result is mismatched")
        self.read_paths.add(path)
        return path, content

    @staticmethod
    def _require_complete_page(data: dict[str, Any]) -> None:
        if data.get("has_more") is not False:
            raise EvidenceError("tool evidence pagination is incomplete")

    def _move_next_or_manifest(self) -> ModelReply:
        if self._pending_paths:
            source = self._pending_paths.pop(0)
            destination = f"archive/{PurePosixPath(source).name}"
            return self._call(
                "move_file",
                {"source": source, "destination": destination},
            )
        names = sorted(PurePosixPath(path).name for path in self.successful_moves)
        self.generated_content = "".join(f"- {name}\n" for name in names)
        self._phase = "archive-write"
        return self._call(
            "write_file",
            {
                "path": "archive/MANIFEST.md",
                "content": self.generated_content,
                "overwrite": False,
            },
        )

    @staticmethod
    def _frontmatter_status(content: str) -> str:
        lines = content.splitlines()
        if not lines or lines[0] != "---":
            raise EvidenceError("tool evidence draft frontmatter is missing")
        try:
            end = lines.index("---", 1)
        except ValueError:
            raise EvidenceError("tool evidence draft frontmatter is invalid") from None
        statuses = [
            line.split(":", 1)[1].strip()
            for line in lines[1:end]
            if line.startswith("status:")
        ]
        if len(statuses) != 1 or statuses[0] not in {"active", "obsolete"}:
            raise EvidenceError("tool evidence draft status is invalid")
        return statuses[0]

    def _build_falcon_index(self) -> str:
        dated: list[tuple[str, str, str]] = []
        formal_name = ""
        for path, content in self._documents.items():
            date_match = re.search(r"(?m)^date: (\d{4}-\d{2}-\d{2})$", content)
            if date_match is None:
                raise EvidenceError("tool evidence Falcon date is missing")
            if (
                "started under the working name Project Falcon" in content
                and "initial scope" in content
            ):
                summary = "项目正式启动并确定初始目标。"
            elif "depends on completion of the vendor security review" in content:
                summary = "项目仍依赖供应商安全评审。"
            elif "current formal project name is Aquila" in content:
                formal_name = "Aquila"
                summary = "最新会议将正式名称更新为 Aquila。"
            else:
                raise EvidenceError("tool evidence Falcon content is unsupported")
            dated.append((date_match.group(1), path, summary))
        if formal_name != "Aquila" or len(dated) != 3:
            raise EvidenceError("tool evidence Falcon set is incomplete")
        sections = [f"当前正式名称：{formal_name}"]
        for date, path, summary in sorted(dated):
            sections.append(f"## {date[:7]}\n\n- {path} — {summary}")
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _validate_write(
        data: dict[str, Any],
        expected_path: str,
        content: str,
    ) -> None:
        if data != {
            "path": expected_path,
            "bytes": len(content.encode("utf-8")),
        }:
            raise EvidenceError("tool evidence write result is mismatched")

    def _call(self, name: str, arguments: dict[str, Any]) -> ModelReply:
        self._counter += 1
        call = ToolCall(
            id=f"{self.task}-{self._counter}",
            name=name,
            arguments=arguments,
        )
        self._expected = call
        return ModelReply(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            },
            tool_calls=(call,),
        )

    @staticmethod
    def _final(content: str) -> ModelReply:
        return ModelReply(message={"role": "assistant", "content": content})
