"""Streaming tool executor — execute tools as their definitions arrive.

Ported from Claude-code's ``StreamingToolExecutor`` pattern.

In the standard LangGraph agent loop, all tool calls are accumulated after
the full LLM response, then executed sequentially.  This executor allows
tools to start executing as soon as their ``tool_use`` blocks arrive in the
stream, parallelising tool execution with response generation.

Concurrency rules (from tool policies):
- ``parallel_safe`` tools may run concurrently with other parallel_safe tools.
- ``serialized`` tools run one at a time, acquiring an async lock.

Results are collected in tool_call order (not completion order) to ensure
deterministic message sequences.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage

from react_agent.execution.policies import ToolPolicy

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of a single tool execution with timing metadata."""

    tool_call_id: str
    tool_name: str
    message: ToolMessage
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


class StreamingToolExecutor:
    """Execute tools concurrently as their definitions arrive during streaming.

    Usage::

        executor = StreamingToolExecutor(tools_by_name, policies)

        # As tool_use blocks arrive from the stream:
        for tool_call in ai_message.tool_calls:
            await executor.submit(tool_call)

        # After the full response:
        results = await executor.collect_results()
    """

    def __init__(
        self,
        tools_by_name: dict[str, Any],
        policies: dict[str, ToolPolicy],
    ) -> None:
        self._tools = tools_by_name
        self._policies = policies
        self._serialized_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Task[ToolExecutionResult]] = {}
        self._results: list[ToolExecutionResult] = []
        self._submission_order: list[str] = []

    async def submit(self, tool_call: dict[str, Any]) -> None:
        """Start executing a tool call immediately.

        For ``serialized`` tools, acquires the serialized lock before
        execution.  For ``parallel_safe`` tools, fires immediately as a
        background task.
        """
        call_id = tool_call.get("id", "")
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        self._submission_order.append(call_id)
        policy = self._policies.get(name, ToolPolicy(name=name))

        if policy.concurrency_class == "serialized":
            # Run serialized tools inline (under lock) and store result immediately
            result = await self._execute_serialized(call_id, name, args, policy)
            self._results.append(result)
        else:
            # Fire parallel_safe tools as background tasks
            task = asyncio.create_task(
                self._execute_one(call_id, name, args, policy),
                name=f"tool-{name}-{call_id[:8]}",
            )
            self._pending[call_id] = task

    async def collect_results(self) -> list[ToolExecutionResult]:
        """Wait for all pending tasks and return results in submission order."""
        # Gather all pending parallel tasks
        for call_id, task in self._pending.items():
            try:
                result = await task
                self._results.append(result)
            except asyncio.CancelledError:
                self._results.append(
                    ToolExecutionResult(
                        tool_call_id=call_id,
                        tool_name="unknown",
                        message=ToolMessage(
                            content="Tool execution was cancelled.",
                            tool_call_id=call_id,
                        ),
                        success=False,
                        error="cancelled",
                    )
                )
            except Exception as exc:
                self._results.append(
                    ToolExecutionResult(
                        tool_call_id=call_id,
                        tool_name="unknown",
                        message=ToolMessage(
                            content=f"Tool execution failed: {exc!s}",
                            tool_call_id=call_id,
                        ),
                        success=False,
                        error=str(exc)[:200],
                    )
                )

        self._pending.clear()

        # Sort by submission order
        order_map = {cid: i for i, cid in enumerate(self._submission_order)}
        self._results.sort(key=lambda r: order_map.get(r.tool_call_id, len(order_map)))

        return list(self._results)

    @property
    def total_tool_duration_ms(self) -> float:
        """Total wall-clock time spent on tool execution across all tools."""
        return sum(r.duration_ms for r in self._results)

    async def _execute_serialized(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        policy: ToolPolicy,
    ) -> ToolExecutionResult:
        """Execute a serialized tool under the shared lock."""
        async with self._serialized_lock:
            return await self._execute_one(call_id, name, args, policy)

    async def _execute_one(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        policy: ToolPolicy,
    ) -> ToolExecutionResult:
        """Execute a single tool with timeout and error handling."""
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                message=ToolMessage(
                    content=f"Tool '{name}' not found.",
                    tool_call_id=call_id,
                ),
                success=False,
                error=f"Tool '{name}' not found",
            )

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(args),
                timeout=policy.timeout_seconds,
            )

            # Normalize result to string
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                import json

                content = json.dumps(result, default=str)
            else:
                content = str(result)

            duration = (time.monotonic() - start) * 1000

            logger.debug(
                "Tool %s completed in %.1fms (result_len=%d)",
                name,
                duration,
                len(content),
            )

            return ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                message=ToolMessage(content=content, tool_call_id=call_id),
                duration_ms=duration,
                success=True,
            )

        except TimeoutError:
            duration = (time.monotonic() - start) * 1000
            logger.warning("Tool %s timed out after %.1fms", name, duration)
            return ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                message=ToolMessage(
                    content=f"Tool '{name}' timed out after {policy.timeout_seconds}s.",
                    tool_call_id=call_id,
                ),
                duration_ms=duration,
                success=False,
                error=f"timeout after {policy.timeout_seconds}s",
            )

        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            logger.warning("Tool %s failed after %.1fms: %s", name, duration, exc)
            return ToolExecutionResult(
                tool_call_id=call_id,
                tool_name=name,
                message=ToolMessage(
                    content=f"Tool '{name}' failed: {exc!s}",
                    tool_call_id=call_id,
                ),
                duration_ms=duration,
                success=False,
                error=str(exc)[:200],
            )
