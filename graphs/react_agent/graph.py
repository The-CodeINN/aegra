"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.

Memory architecture (LangMem):
  Short-term  — SummarizationNode runs before every call_model turn. When the
                message history exceeds _SUMMARY_TRIGGER_TOKENS, older messages
                are compressed into a running summary and replaced.  The model
                always operates within the _TRIM_MAX_TOKENS budget.

  Long-term   — create_manage_memory_tool / create_search_memory_tool give the
                agent native tools to persist and retrieve durable facts about
                the user across ALL threads.  The agent decides proactively when
                to store or search — no keyword heuristics are required.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langmem import create_manage_memory_tool, create_memory_store_manager, create_search_memory_tool
from langmem.short_term import SummarizationNode

from react_agent import prompts as _prompts
from react_agent.compaction import (
    CompactionTier,
    context_collapse,
    microcompact_messages,
    select_compaction_tier,
)
from react_agent.context import Context
from react_agent.cost_tracker import SessionCost, is_over_cost_threshold, track_llm_usage
from react_agent.execution import ExecutionEvent, build_runtime_tools, build_tool_limit_notice, serialize_event
from react_agent.guardrails import (
    INJECTION_BLOCKED_RESPONSE,
    LEAK_SAFE_RESPONSE,
    screen_input_for_injection,
    screen_output_for_hallucination,
    screen_output_for_leak,
)
from react_agent.memory import DEFAULT_MEMORY_NAMESPACE, MAX_MEMORIES_PER_USER, MEMORY_SCHEMAS, memory_freshness_note
from react_agent.message_utils import (
    apply_tool_result_budget as _apply_tool_result_budget,
)
from react_agent.message_utils import (
    sanitize_messages_for_anthropic as _sanitize_messages,
)
from react_agent.message_utils import (
    snip_old_tool_results as _snip_old_tool_results,
)
from react_agent.retry import CannotRetryError, with_retry
from react_agent.sanitization import sanitize_unicode as _sanitize_unicode
from react_agent.sanitized_anthropic import SanitizedChatAnthropic
from react_agent.session_memory import (
    SESSION_MEMORY_TEMPLATE,
    build_session_context_block,
    extract_session_memory,
    get_session_namespace,
    should_extract_session_memory,
)
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, load_chat_model

logger = logging.getLogger(__name__)

# Short-term memory token budgets (LangMem SummarizationNode)
_TRIM_MAX_TOKENS = 8000  # max tokens returned to call_model each turn
_SUMMARY_TRIGGER_TOKENS = 6000  # summarise when history exceeds this
_MAX_SUMMARY_TOKENS = 512  # budget for the generated summary itself

# Module-level cache: one SummarizationNode per model string to avoid
# rebuilding the LLM client on every graph turn.
_summarization_node_cache: dict[str, SummarizationNode] = {}

# Fallback model map: primary model → cheaper fallback for overload recovery.
# Ported from Claude-code's fallback model system.
# Bedrock-only (EU cross-region inference profiles + Kimi K2.5).
_FALLBACK_MODELS: dict[str, str] = {
    "bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0": "bedrock/moonshotai.kimi-k2.5",
    "bedrock/moonshotai.kimi-k2.5": "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Max output token recovery: retry count when model hits max_tokens stop reason.
_MAX_OUTPUT_RECOVERY_RETRIES = 3


class _StringContentModel:
    """Thin wrapper around a chat model that normalises `.content` to a plain string.

    Anthropic and Bedrock both return `AIMessage.content` as a list of content
    blocks (e.g. ``[{'type': 'text', 'text': '...'}]``).  LangMem's
    ``SummarizationNode`` stores the raw ``.content`` value directly in
    ``RunningSummary.summary`` which is declared as ``str``.  When the summary
    is later injected into the next context window (via ``DEFAULT_FINAL_SUMMARY_PROMPT``
    / ``DEFAULT_EXISTING_SUMMARY_PROMPT``), Python's ``str.format()`` calls
    ``str()`` on the list − producing the Python-repr form
    ``[{'type': 'text', 'text': '...'}]`` with single quotes.  That repr leaks
    into the agent's visible system context and ultimately into streamed output.

    This wrapper intercepts the model response and replaces list content with
    the extracted plain text before returning, so ``RunningSummary.summary``
    is always a ``str``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def _normalise(self, response: Any) -> Any:
        """Replace list content with extracted plain text."""
        if isinstance(getattr(response, "content", None), list):
            response.content = get_message_text(response)
        return response

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._normalise(self._model.invoke(messages, **kwargs))

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return self._normalise(await self._model.ainvoke(messages, **kwargs))

    def __getattr__(self, name: str) -> Any:  # forward everything else
        return getattr(self._model, name)


_COHERE_MAX_CHARS = 1800  # Cohere embedding API hard limit is 2048 chars; leave headroom


def _truncate_for_embedding(text: str) -> str:
    """Truncate *text* to fit within Cohere's embedding character limit."""
    if len(text) <= _COHERE_MAX_CHARS:
        return text
    return text[:_COHERE_MAX_CHARS]


def _normalize_messages_for_memory(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Return a copy of *messages* where every AIMessage has plain-string content.

    ``langmem.utils.get_conversation()`` calls ``msg.pretty_repr()`` on each
    message.  For AIMessages whose ``.content`` is a list of content blocks,
    ``pretty_repr()`` renders the Python repr of that list (single quotes,
    dict format) rather than the actual text.  This raw repr then appears
    inside the ``<session_…>`` XML tags sent to the memory-extraction LLM and
    can bleed into streamed output when the agent echoes context back.

    Converting list content to a plain string here ensures that only readable
    text flows into the memory pipeline.  All message texts are also truncated
    to stay within Cohere's 2048-character embedding limit.
    """
    normalized: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            text = _truncate_for_embedding(get_message_text(msg))
            new_msg = AIMessage(
                content=text,
                id=msg.id,
                tool_calls=getattr(msg, "tool_calls", []),
                response_metadata=getattr(msg, "response_metadata", {}),
            )
            normalized.append(new_msg)
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str):
            truncated = _truncate_for_embedding(msg.content)
            if truncated != msg.content:
                new_msg = AIMessage(
                    content=truncated,
                    id=msg.id,
                    tool_calls=getattr(msg, "tool_calls", []),
                    response_metadata=getattr(msg, "response_metadata", {}),
                )
                normalized.append(new_msg)
            else:
                normalized.append(msg)
        elif isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            # Frontend sends multimodal blocks: [{'type': 'text', 'text': '...'}]
            # get_message_text extracts the plain text from these blocks
            text = _truncate_for_embedding(get_message_text(msg))
            normalized.append(HumanMessage(content=text, id=msg.id))
        elif isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            truncated = _truncate_for_embedding(msg.content)
            if truncated != msg.content:
                normalized.append(HumanMessage(content=truncated, id=msg.id))
            else:
                normalized.append(msg)
        else:
            # ToolMessages and other types — truncate content to stay within
            # Cohere's 2048-character embedding limit.
            if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 1900:
                from copy import copy  # noqa: PLC0415

                truncated_msg = copy(msg)
                truncated_msg.content = _truncate_for_embedding(msg.content)
                normalized.append(truncated_msg)
            else:
                normalized.append(msg)
    return normalized


def _is_anthropic_model(model_name: str) -> bool:
    return model_name.split("/", maxsplit=1)[0].lower() == "anthropic"


def _is_bedrock_model(model_name: str) -> bool:
    return model_name.split("/", maxsplit=1)[0].lower() == "bedrock"


def _extract_provider_model(model_name: str) -> str:
    return model_name.split("/", maxsplit=1)[1]


def _get_summarization_node(runtime: Runtime[Context]) -> SummarizationNode:
    """Return a cached SummarizationNode for the current model.

    Nodes are cached by model name so the LLM client is only built once per
    unique model string, not on every graph turn.
    """
    model_key = runtime.context.model
    if model_key not in _summarization_node_cache:
        # Wrap with _StringContentModel so that RunningSummary.summary is always a
        # plain str.  Without this, Anthropic/Bedrock returns content blocks
        # ([{'type':'text','text':'...'}]) which LangMem stores verbatim; the list
        # is later str()-formatted into the summary SystemMessage, producing the
        # Python repr with single quotes that leaks into the agent's context.
        model = _StringContentModel(_build_runtime_model(runtime))
        _summarization_node_cache[model_key] = SummarizationNode(
            model=model,
            max_tokens=_TRIM_MAX_TOKENS,
            max_tokens_before_summary=_SUMMARY_TRIGGER_TOKENS,
            max_summary_tokens=_MAX_SUMMARY_TOKENS,
            token_counter=count_tokens_approximately,
            input_messages_key="messages",
            output_messages_key="summarized_messages",
        )
    return _summarization_node_cache[model_key]


def _build_langmem_tools(runtime: Runtime[Context]) -> list[Any]:
    """Return LangMem manage_memory + search_memory tools scoped to this user.

    Each tool is scoped to ``(user_id, "memories")`` in the runtime store so
    memories from different users never bleed into each other.  Returns an
    empty list when the store or user_id is unavailable.
    """
    store = runtime.store
    user_id = runtime.context.user_id
    if not store or not user_id:
        return []
    namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
    return [
        create_manage_memory_tool(
            namespace=namespace,
            store=store,
            actions_permitted=("create", "update", "delete"),
        ),
        create_search_memory_tool(namespace=namespace, store=store),
    ]


def _resolve_runtime_tooling(runtime: Runtime[Context]) -> tuple[list[Any], dict[str, Any]]:
    """Return provider-ready runtime tools plus the effective policy registry."""

    all_tools: list[Any] = list(TOOLS) + _build_langmem_tools(runtime)
    return build_runtime_tools(
        tools=all_tools,
        model_name=runtime.context.model,
        anthropic_tool_search_enabled=runtime.context.anthropic_tool_search_enabled,
        anthropic_tool_search_variant=runtime.context.anthropic_tool_search_variant,
        anthropic_programmatic_tool_calling_enabled=runtime.context.anthropic_programmatic_tool_calling_enabled,
    )


def _build_runtime_model(runtime: Runtime[Context]) -> Any:
    if _is_bedrock_model(runtime.context.model):
        return load_chat_model(
            runtime.context.model,
            enable_thinking=runtime.context.enable_thinking,
            thinking_budget=runtime.context.thinking_budget,
        )

    if _is_anthropic_model(runtime.context.model):
        model_kwargs: dict[str, Any] = {
            "model": _extract_provider_model(runtime.context.model),
            "temperature": 0,
        }

        betas: list[str] = []
        if runtime.context.anthropic_programmatic_tool_calling_enabled:
            betas.append("advanced-tool-use-2025-11-20")
            model_kwargs["reuse_last_container"] = True

        if betas:
            model_kwargs["betas"] = betas

        return SanitizedChatAnthropic(**model_kwargs)

    return load_chat_model(
        runtime.context.model,
        enable_thinking=runtime.context.enable_thinking,
        thinking_budget=runtime.context.thinking_budget,
    )


def _build_runtime_middleware(runtime: Runtime[Context]) -> list[Any]:
    if _is_bedrock_model(runtime.context.model):
        # Bedrock handles caching at the infrastructure level; no client middleware needed.
        return []
    if _is_anthropic_model(runtime.context.model) and runtime.context.anthropic_prompt_caching_enabled:
        return [AnthropicPromptCachingMiddleware(ttl=runtime.context.anthropic_prompt_caching_ttl)]
    return []


def _build_runtime_tools(runtime: Runtime[Context]) -> list[Any]:
    runtime_tools, _ = _resolve_runtime_tooling(runtime)
    return runtime_tools


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return get_message_text(message).strip()
    return ""


async def _invoke_integrated_agent(
    messages: list[AnyMessage], runtime: Runtime[Context], system_message: str
) -> tuple[AIMessage, list[dict[str, Any]], dict[str, Any], list[str]]:
    """Invoke a single integrated LangChain agent runtime for all providers.

    Returns:
        (ai_message, execution_events, usage_metadata, tool_result_texts) —
        the final AI response, any execution events emitted during invocation,
        token usage metadata from the LLM response for cost tracking, and
        the text content of all ToolMessages from the agent's internal
        tool-calling loop (used for hallucination grounding).
    """
    runtime_tools, _ = _resolve_runtime_tooling(runtime)
    agent = create_agent(
        model=_build_runtime_model(runtime),
        tools=runtime_tools,
        middleware=_build_runtime_middleware(runtime),
        system_prompt=system_message,
    )

    # Always normalise frontend multimodal content blocks (image/file/text-plain) into
    # the provider-specific format, and strip Anthropic beta incompatibilities from AIMessages.
    provider = "bedrock" if _is_bedrock_model(runtime.context.model) else "anthropic"
    messages = _sanitize_messages(messages, provider=provider)

    @with_retry(max_retries=3)
    async def _run() -> dict[str, Any]:
        return await agent.ainvoke({"messages": messages})

    try:
        result = await _run()
    except CannotRetryError as exc:
        # Check if we should try a fallback model (overload scenario)
        if getattr(exc, "context", None) and getattr(exc.context, "last_status_code", None) == 529:
            fallback_model = _FALLBACK_MODELS.get(runtime.context.model)
            if fallback_model:
                logger.warning(
                    "Primary model overloaded (529) — falling back to %s",
                    fallback_model,
                )
                # Temporarily swap model for fallback invocation
                original_model = runtime.context.model
                runtime.context.model = fallback_model
                try:
                    fallback_agent = create_agent(
                        model=_build_runtime_model(runtime),
                        tools=runtime_tools,
                        middleware=_build_runtime_middleware(runtime),
                        system_prompt=system_message,
                    )

                    @with_retry(max_retries=2)
                    async def _run_fallback() -> dict[str, Any]:
                        return await fallback_agent.ainvoke({"messages": messages})

                    result = await _run_fallback()

                    final_messages = result.get("messages", [])
                    fb_tool_texts = [
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                        for msg in final_messages
                        if isinstance(msg, ToolMessage)
                    ]
                    for msg in reversed(final_messages):
                        if isinstance(msg, AIMessage):
                            usage = getattr(msg, "response_metadata", {}).get("usage", {})
                            events = [
                                serialize_event(
                                    ExecutionEvent(
                                        event_type="fallback_model_activated",
                                        level="warning",
                                        message=f"Switched to fallback model {fallback_model} after primary overload.",
                                        metadata={
                                            "primary_model": original_model,
                                            "fallback_model": fallback_model,
                                        },
                                    )
                                )
                            ]
                            return msg, events, usage, fb_tool_texts
                finally:
                    # Restore original model
                    runtime.context.model = original_model

        logger.exception("Agent invocation failed after retries")
        events = [
            serialize_event(
                ExecutionEvent(
                    event_type="retry_exhausted",
                    level="error",
                    message="Model invocation failed after retry exhaustion.",
                    metadata={"component": "model"},
                )
            ),
            serialize_event(
                ExecutionEvent(
                    event_type="fallback_mode",
                    level="warning",
                    message="Returned fallback model response after retry exhaustion.",
                    metadata={"component": "model"},
                )
            ),
        ]
        return (
            AIMessage(content="I'm having trouble reaching my AI model right now. Please try again in a moment."),
            events,
            {},
            [],
        )

    final_messages = result.get("messages", [])
    agent_tool_texts = [
        msg.content if isinstance(msg.content, str) else str(msg.content)
        for msg in final_messages
        if isinstance(msg, ToolMessage)
    ]
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "response_metadata", {}).get("usage", {})
            return msg, [], usage, agent_tool_texts

    return (
        AIMessage(content="I was unable to produce a response from the agent runtime."),
        [
            serialize_event(
                ExecutionEvent(
                    event_type="fallback_mode",
                    level="warning",
                    message="Agent runtime returned no AIMessage; fallback response emitted.",
                    metadata={"component": "model"},
                )
            )
        ],
        {},
        [],
    )


async def _load_session_notes_into_state(result: dict[str, Any], runtime: Runtime[Context]) -> None:
    """Load persisted session notes from the store into the state dict.

    Called from ``screen_input`` so that ``call_model`` has access to session
    notes for injection into the system prompt on every turn — not just on
    turns where extraction runs.
    """
    store = runtime.store
    user_id = runtime.context.user_id
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    if not store or not user_id or not thread_id:
        return
    try:
        session_ns = get_session_namespace(user_id, thread_id)
        existing = await store.asearch(session_ns, limit=1)
        if existing:
            notes = existing[0].value.get("notes", "")
            if notes:
                result["session_notes"] = notes
    except Exception:
        logger.debug("Failed to load session notes from store.", exc_info=True)


async def screen_input(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Pre-screen the latest user message for prompt-injection / jailbreak attempts.

    Also normalises ``HumanMessage.content`` from the frontend's multimodal
    content-block format (``[{'type': 'text', 'text': '...'}]``) to a plain
    string.  Without this, every subsequent ``values`` stream event re-emits
    the full state containing the list-format content, which the frontend
    briefly renders as a raw Python repr — causing a visible flicker.

    If guardrails are disabled or no injection is detected, returns an empty
    dict (plus any normalised messages) so the graph proceeds to ``summarize``
    unchanged.

    If an injection is detected, adds a safe refusal ``AIMessage`` and sets
    ``guardrail_blocked=True`` so the graph short-circuits to ``__end__``.
    """
    # Normalise HumanMessage list content → plain string so every values event
    # emitted by downstream nodes (generate_thread_title, consolidate_memories)
    # contains clean renderable content.
    normalized_messages: list[AnyMessage] = []
    needs_normalization = False
    for msg in state.messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            text = get_message_text(msg)
            normalized_messages.append(HumanMessage(content=text, id=msg.id))
            needs_normalization = True
        else:
            normalized_messages.append(msg)

    if not runtime.context.guardrails_enabled:
        result: dict[str, Any] = {"messages": normalized_messages} if needs_normalization else {}
        # Load session notes from store if available (so call_model has them)
        await _load_session_notes_into_state(result, runtime)
        return result

    user_text = _latest_human_text(list(state.messages))
    if not user_text:
        result = {"messages": normalized_messages} if needs_normalization else {}
        await _load_session_notes_into_state(result, runtime)
        return result

    # Strip invisible/directional Unicode characters that could be used for
    # prompt injection or bi-directional text spoofing before classification.
    user_text = _sanitize_unicode(user_text)

    guardrail_model = load_chat_model(runtime.context.guardrail_model)
    is_injection = await screen_input_for_injection(user_text, guardrail_model)

    if not is_injection:
        result = {"messages": normalized_messages} if needs_normalization else {}
        await _load_session_notes_into_state(result, runtime)
        return result

    logger.warning(
        "Injection blocked — user_id=%s thread input screened and rejected",
        runtime.context.user_id,
        extra={"user_id": runtime.context.user_id, "input_preview": user_text[:120]},
    )
    result: dict[str, Any] = {
        "messages": [AIMessage(content=INJECTION_BLOCKED_RESPONSE)],
        "guardrail_blocked": True,
    }
    return result


def route_after_screening(state: State) -> Literal["summarize", "__end__"]:
    """Route to ``__end__`` when an injection was blocked, otherwise to ``summarize``."""
    return "__end__" if state.guardrail_blocked else "summarize"


_SUMMARIZATION_CIRCUIT_BREAKER_LIMIT = 3


async def summarize(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Short-term memory node — compress long message history with LangMem.

    Uses LangMem's ``SummarizationNode`` which:
    - Passes messages through unchanged when history is short.
    - Once the token count exceeds ``_SUMMARY_TRIGGER_TOKENS``, compresses
      older messages into a concise summary message at the front of the list.
    - Tracks already-summarised message IDs in the ``RunningSummary`` stored
      under ``state.context`` to avoid re-summarising the same messages.

    The resulting ``summarized_messages`` is what ``call_model`` sends to the
    LLM, always within the ``_TRIM_MAX_TOKENS`` budget.

    Circuit breaker: after ``_SUMMARIZATION_CIRCUIT_BREAKER_LIMIT`` consecutive
    failures the node stops attempting compression (prevents the 3,000+
    failure-per-session spiral observed in production).
    """
    if state.summarization_failure_count >= _SUMMARIZATION_CIRCUIT_BREAKER_LIMIT:
        logger.warning(
            "Summarization circuit breaker open (%d consecutive failures) — passing full message list to call_model",
            state.summarization_failure_count,
        )
        return {
            "summarized_messages": list(state.messages),
            "summarization_failure_count": state.summarization_failure_count,
        }

    try:
        node = _get_summarization_node(runtime)
        result = await node.ainvoke(
            {"messages": list(state.messages), "context": dict(state.context or {})},
            config=RunnableConfig(callbacks=[]),
        )
        # Reset failure count on success
        result["summarization_failure_count"] = 0
        return result
    except Exception:
        new_count = state.summarization_failure_count + 1
        logger.exception(
            "Summarization failed (failure %d/%d) — passing full message list to call_model",
            new_count,
            _SUMMARIZATION_CIRCUIT_BREAKER_LIMIT,
        )
        # Passthrough on failure so call_model always has something to work with.
        return {
            "summarized_messages": list(state.messages),
            "summarization_failure_count": new_count,
        }


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Call the LLM powering the agent.

    Uses ``state.summarized_messages`` (produced by the ``summarize`` node) as
    the message window so the model always operates within its token budget.

    Long-term memory is handled entirely by the agent's own ``manage_memory``
    and ``search_memory`` tools (LangMem hot path).  No manual extraction,
    keyword detection, or system-prompt injection is required here.

    Enhanced with:
    - Fallback model support (automatic model swap on overload)
    - Multi-tier compaction (microcompact + context collapse)
    - Hallucination screening (grounding verification)
    - Cost/token tracking per session
    - Max output token recovery
    """
    # Use str.replace instead of .format() — the 1000-line prompt contains literal
    # {blocks} in examples/directives; .format() would KeyError on any unknown placeholder.
    static_prompt = runtime.context.system_prompt.replace("{system_time}", datetime.now(tz=UTC).isoformat())

    _, tool_policies = _resolve_runtime_tooling(runtime)
    execution_events = list(state.execution_events)
    session_cost = state.session_cost or SessionCost()
    tool_limit_notice = build_tool_limit_notice(state.tool_call_counts, tool_policies)
    if tool_limit_notice:
        execution_events.append(
            serialize_event(
                ExecutionEvent(
                    event_type="tool_limit_notice",
                    message="One or more tools reached the configured call limit.",
                    metadata={
                        "tool_call_counts": dict(state.tool_call_counts),
                    },
                )
            )
        )

    # Assemble the final system prompt via the section builder.  Static sections
    # (memory instructions) are memoized across turns; volatile sections
    # (tool_limit_notice) are appended fresh each turn.
    system_message = _prompts.build_runtime_system_prompt(
        static_prompt,
        tool_limit_notice=tool_limit_notice,
    )

    # --- Session context injection ---
    # If we have session notes from a previous extraction, inject them into
    # the system prompt so the agent retains context after compaction.
    session_block = build_session_context_block(state.session_notes)
    if session_block:
        system_message = system_message + "\n" + session_block

    # --- Proactive memory load ---
    # On the first turn (no previous AI messages), pre-load relevant memories
    # and inject them as a system reminder.  This ensures the agent has context
    # before its first response without relying on it to call search_memory.
    proactive_memory_text: str = ""
    store = runtime.store
    user_id = runtime.context.user_id
    if store and user_id:
        ai_count = sum(1 for m in state.messages if isinstance(m, AIMessage))
        if ai_count == 0:
            try:
                namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
                existing_memories = await store.asearch(namespace, limit=20)
                if existing_memories:
                    memory_lines: list[str] = []
                    for item in existing_memories:
                        kind = item.value.get("kind", "unknown")
                        content = item.value.get("content", {})
                        updated_at = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
                        freshness = memory_freshness_note(updated_at)
                        summary = str(content) if isinstance(content, dict) else str(content)
                        line = f"  [{kind}] {summary[:200]}"
                        if freshness:
                            line += f"\n  {freshness}"
                        memory_lines.append(line)
                    if memory_lines:
                        memory_block = "\n".join(memory_lines)
                        proactive_memory_text = memory_block
                        # Strip freshness metadata from what the model sees —
                        # the <system-reminder> tags cause the model to make
                        # false temporal claims like "it's been a few days".
                        clean_block = re.sub(
                            r"\n?\s*<system-reminder>.*?</system-reminder>",
                            "",
                            memory_block,
                            flags=re.DOTALL,
                        )
                        system_message += f"""
<proactive_memory_recall>
The following memories were recalled for this student at the start of the conversation.
Use them to personalize your response. Do NOT repeat these verbatim — weave them naturally.
IMPORTANT: Do NOT make claims about when you last spoke, how long it has been, or when
the student enrolled. You do not know this information. Treat every new thread as a fresh
conversation and greet the student without time references.

{clean_block}
</proactive_memory_recall>"""
            except Exception:
                logger.debug("Proactive memory load failed; continuing without.", exc_info=True)

    # Use the token-bounded summarised messages from the previous node.
    # Fall back to the full message list on the very first turn (before
    # summarize has had a chance to produce output).
    prepared_messages = list(state.summarized_messages) or list(state.messages)

    # Budget tool results: truncate oversized individual results, then clear
    # old ones to free context space.  Applied before every model call so the
    # effective prompt never grows unboundedly with accumulated tool output.
    prepared_messages = _apply_tool_result_budget(prepared_messages)
    prepared_messages = _snip_old_tool_results(prepared_messages)

    # --- Multi-tier compaction check (Tier 1: microcompact) ---
    _did_microcompact = False
    if not state.has_attempted_microcompact:
        tier = select_compaction_tier(
            prepared_messages,
            has_attempted_microcompact=state.has_attempted_microcompact,
            has_attempted_collapse=state.has_attempted_context_collapse,
        )
        if tier == CompactionTier.MICROCOMPACT:
            try:
                compact_model = load_chat_model(runtime.context.guardrail_model)
                prepared_messages, chars_saved = await microcompact_messages(prepared_messages, compact_model)
                if chars_saved > 0:
                    _did_microcompact = True
                    execution_events.append(
                        serialize_event(
                            ExecutionEvent(
                                event_type="microcompact_applied",
                                message=f"Microcompact saved {chars_saved} characters.",
                                metadata={"chars_saved": chars_saved},
                            )
                        )
                    )
            except Exception:
                logger.warning("Microcompact failed — continuing with original messages")

    # Reactive compact: if the context is too long and we haven't tried yet,
    # force a summarisation pass and retry the model call once.
    # Mirrors Claude-code's REACTIVE_COMPACT pattern.
    _did_reactive_compact = False
    _did_context_collapse = False
    agent_tool_results: list[str] = []
    try:
        response, invoke_events, usage_metadata, agent_tool_results = await _invoke_integrated_agent(
            prepared_messages, runtime, system_message
        )
        execution_events.extend(invoke_events)
        # Track cost from usage metadata
        if usage_metadata:
            session_cost = track_llm_usage(session_cost, runtime.context.model, usage_metadata)
    except Exception as first_exc:
        exc_str = str(first_exc).lower()
        is_context_overflow = any(
            marker in exc_str
            for marker in (
                "context_length_exceeded",
                "too many tokens",
                "prompt too long",
                "input is too long",
                "context window",
            )
        )
        if is_context_overflow and not state.has_attempted_reactive_compact:
            execution_events.append(
                serialize_event(
                    ExecutionEvent(
                        event_type="reactive_compact_attempted",
                        level="warning",
                        message="Attempting reactive compact after context overflow.",
                        metadata={"error": str(first_exc)[:200]},
                    )
                )
            )
            logger.warning(
                "Context overflow detected — attempting reactive compact before retry: %s",
                str(first_exc)[:200],
            )
            try:
                summ_node = _get_summarization_node(runtime)
                summ_result = await summ_node.ainvoke(
                    {
                        "messages": list(state.messages),
                        "context": dict(state.context or {}),
                    },
                    config=RunnableConfig(callbacks=[]),
                )
                prepared_messages = summ_result.get("summarized_messages", prepared_messages)
                response, invoke_events, usage_metadata, agent_tool_results = await _invoke_integrated_agent(
                    prepared_messages, runtime, system_message
                )
                execution_events.extend(invoke_events)
                if usage_metadata:
                    session_cost = track_llm_usage(session_cost, runtime.context.model, usage_metadata)
                execution_events.append(
                    serialize_event(
                        ExecutionEvent(
                            event_type="reactive_compact_succeeded",
                            message="Reactive compact succeeded and model invocation was retried.",
                        )
                    )
                )
                _did_reactive_compact = True
            except Exception:
                # Reactive compact failed — try context collapse as last resort
                logger.exception("Reactive compact recovery also failed — attempting context collapse")
                if not state.has_attempted_context_collapse:
                    try:
                        collapse_model = load_chat_model(runtime.context.guardrail_model)
                        collapsed = await context_collapse(list(state.messages), collapse_model)
                        response, invoke_events, usage_metadata, agent_tool_results = await _invoke_integrated_agent(
                            collapsed, runtime, system_message
                        )
                        execution_events.extend(invoke_events)
                        if usage_metadata:
                            session_cost = track_llm_usage(session_cost, runtime.context.model, usage_metadata)
                        execution_events.append(
                            serialize_event(
                                ExecutionEvent(
                                    event_type="context_collapse_triggered",
                                    level="warning",
                                    message="Context collapse succeeded after reactive compact failure.",
                                )
                            )
                        )
                        _did_context_collapse = True
                    except Exception:
                        logger.exception("Context collapse also failed")
                        execution_events.append(
                            serialize_event(
                                ExecutionEvent(
                                    event_type="reactive_compact_failed",
                                    level="error",
                                    message="All compaction tiers failed; returning fallback response.",
                                )
                            )
                        )
                        response = AIMessage(
                            content="I'm having trouble processing this conversation right now. Please try again."
                        )
                        execution_events.append(
                            serialize_event(
                                ExecutionEvent(
                                    event_type="fallback_mode",
                                    level="warning",
                                    message="Returned fallback response after all compaction tiers failed.",
                                    metadata={"component": "model"},
                                )
                            )
                        )
                else:
                    execution_events.append(
                        serialize_event(
                            ExecutionEvent(
                                event_type="reactive_compact_failed",
                                level="error",
                                message="Reactive compact failed; returning fallback response.",
                            )
                        )
                    )
                    response = AIMessage(
                        content="I'm having trouble processing this conversation right now. Please try again."
                    )
                    execution_events.append(
                        serialize_event(
                            ExecutionEvent(
                                event_type="fallback_mode",
                                level="warning",
                                message="Returned fallback response after reactive compact failure.",
                                metadata={"component": "model"},
                            )
                        )
                    )
        else:
            raise

    # --- Max output token recovery ---
    # If the model hit its output token limit, retry with a continuation prompt.
    stop_reason = getattr(response, "response_metadata", {}).get("stop_reason", "")
    if stop_reason == "max_tokens" and not response.tool_calls:
        for recovery_attempt in range(_MAX_OUTPUT_RECOVERY_RETRIES):
            execution_events.append(
                serialize_event(
                    ExecutionEvent(
                        event_type="max_output_recovery_attempted",
                        level="warning",
                        message=f"Max output tokens hit — recovery attempt {recovery_attempt + 1}/{_MAX_OUTPUT_RECOVERY_RETRIES}.",
                    )
                )
            )
            continuation_messages = prepared_messages + [
                response,
                HumanMessage(
                    content=(
                        "Your previous response was cut off mid-sentence. "
                        "Continue EXACTLY from where you stopped. "
                        "Do not repeat what you already said — pick up seamlessly."
                    )
                ),
            ]
            try:
                continuation, cont_events, cont_usage, _ = await _invoke_integrated_agent(
                    continuation_messages, runtime, system_message
                )
                execution_events.extend(cont_events)
                if cont_usage:
                    session_cost = track_llm_usage(session_cost, runtime.context.model, cont_usage)

                # Merge the continuation into the original response
                original_text = get_message_text(response)
                continuation_text = get_message_text(continuation)
                response = AIMessage(
                    content=original_text + continuation_text,
                    id=response.id,
                    tool_calls=getattr(continuation, "tool_calls", []),
                    response_metadata=getattr(continuation, "response_metadata", {}),
                )

                cont_stop = getattr(continuation, "response_metadata", {}).get("stop_reason", "")
                if cont_stop != "max_tokens":
                    execution_events.append(
                        serialize_event(
                            ExecutionEvent(
                                event_type="max_output_recovery_succeeded",
                                message=f"Max output recovery succeeded on attempt {recovery_attempt + 1}.",
                            )
                        )
                    )
                    break
            except Exception:
                logger.warning("Max output recovery attempt %d failed", recovery_attempt + 1)
                execution_events.append(
                    serialize_event(
                        ExecutionEvent(
                            event_type="max_output_recovery_failed",
                            level="warning",
                            message=f"Recovery attempt {recovery_attempt + 1} failed.",
                        )
                    )
                )
                break

    # Anthropic/Bedrock return AIMessage.content as a list of content blocks
    # (e.g. [{'type': 'text', 'text': '...'}]).  Normalise to a plain string
    # so that every `values` stream event the frontend receives has clean,
    # renderable content — preventing the raw Python repr from flashing on
    # re-renders triggered by later graph nodes (e.g. generate_thread_title).
    if isinstance(response.content, list):
        response = AIMessage(
            content=get_message_text(response),
            id=response.id,
            tool_calls=getattr(response, "tool_calls", []),
            response_metadata=getattr(response, "response_metadata", {}),
        )

    # Output screening: replace the response if it leaks system-prompt content.
    if runtime.context.guardrails_enabled and screen_output_for_leak(get_message_text(response)):
        execution_events.append(
            serialize_event(
                ExecutionEvent(
                    event_type="output_guardrail_blocked",
                    level="warning",
                    message="Output leak screening replaced the model response.",
                )
            )
        )
        response = AIMessage(content=LEAK_SAFE_RESPONSE, id=response.id)

    # Hallucination screening: check if the response fabricates user facts.
    # Only run on final responses (no tool calls) to avoid screening intermediate steps.
    response_text = get_message_text(response)
    if (
        runtime.context.guardrails_enabled
        and not response.tool_calls
        and response_text
        and response_text != LEAK_SAFE_RESPONSE
    ):
        # Collect tool results from the conversation for evidence grounding.
        # Include both outer-graph ToolMessages (from previous turns) and
        # the agent runtime's internal tool results (current turn) so the
        # hallucination checker sees all evidence the model actually used.
        tool_result_texts = [
            msg.content if isinstance(msg.content, str) else str(msg.content)
            for msg in state.messages
            if isinstance(msg, ToolMessage)
        ] + agent_tool_results
        # Include proactive memory content as evidence — on the first turn
        # the model personalizes from recalled memories, not tool calls.
        if proactive_memory_text:
            tool_result_texts.append(proactive_memory_text)
        # Include session notes as evidence — these are injected into the
        # system prompt and the model may reference them.
        if session_block:
            tool_result_texts.append(session_block)
        user_text = _latest_human_text(list(state.messages))

        guardrail_model = load_chat_model(runtime.context.guardrail_model)
        is_hallucination, hallucination_details = await screen_output_for_hallucination(
            response_text, tool_result_texts, user_text, guardrail_model
        )
        if is_hallucination:
            execution_events.append(
                serialize_event(
                    ExecutionEvent(
                        event_type="hallucination_detected",
                        level="warning",
                        message="Hallucination screening detected fabricated user facts.",
                        metadata={
                            "response_preview": response_text[:200],
                            "details": hallucination_details,
                        },
                    )
                )
            )
            # Log only — do not modify the response shown to the user.
            # The guardrail is still too aggressive with proactive-memory
            # paraphrases, so we keep it as an observability signal while
            # we tune precision.

    # Cost threshold warning
    if is_over_cost_threshold(session_cost):
        execution_events.append(
            serialize_event(
                ExecutionEvent(
                    event_type="cost_threshold_warning",
                    level="warning",
                    message=f"Session cost ${session_cost.total_cost_usd:.4f} exceeds threshold ${session_cost.cost_warning_threshold_usd:.2f}.",
                    metadata={
                        "total_cost_usd": session_cost.total_cost_usd,
                        "threshold_usd": session_cost.cost_warning_threshold_usd,
                    },
                )
            )
        )

    # Tally tool calls from the full message history so tool_call_counts state
    # reflects cumulative usage across turns for the limit enforcement above.
    updated_counts: dict[str, int] = {}
    for msg in state.messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []) or []:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name:
                    updated_counts[name] = updated_counts.get(name, 0) + 1

    # --- Mutual exclusion: detect if the agent called manage_memory ---
    # If the agent wrote memories during the hot path, consolidate_memories
    # should skip to avoid duplication (Claude-code's mutual exclusion pattern).
    _agent_wrote_memory = updated_counts.get("manage_memory", 0) > 0

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ],
            "tool_call_counts": updated_counts,
            "has_attempted_reactive_compact": state.has_attempted_reactive_compact or _did_reactive_compact,
            "has_attempted_context_collapse": state.has_attempted_context_collapse or _did_context_collapse,
            "has_attempted_microcompact": state.has_attempted_microcompact or _did_microcompact,
            "session_cost": session_cost,
            "execution_events": execution_events,
            "agent_wrote_memory": _agent_wrote_memory,
        }

    return {
        "messages": [response],
        "tool_call_counts": updated_counts,
        "has_attempted_reactive_compact": state.has_attempted_reactive_compact or _did_reactive_compact,
        "has_attempted_context_collapse": state.has_attempted_context_collapse or _did_context_collapse,
        "has_attempted_microcompact": state.has_attempted_microcompact or _did_microcompact,
        "session_cost": session_cost,
        "execution_events": execution_events,
        "agent_wrote_memory": _agent_wrote_memory,
    }


async def generate_thread_title(state: State, runtime: Runtime[Context]) -> dict:
    """Generate a short title for the thread after the first complete exchange.

    Uses the full first exchange (first human message + first AI response) so
    that greetings like "hi" or "hello" still produce a meaningful title drawn
    from what the AI actually talked about.
    """
    human_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in state.messages if isinstance(m, AIMessage)]
    if not human_messages:
        return {}

    first_human = get_message_text(human_messages[0])[:400]
    first_ai = get_message_text(ai_messages[0])[:400] if ai_messages else ""

    exchange = f"User: {first_human}"
    if first_ai:
        exchange += f"\nAssistant: {first_ai}"

    model = load_chat_model(
        # Use Haiku for title generation — cheap, fast, and available in EU.
        "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": (
                    "Generate a concise 4-6 word title for this conversation. "
                    "Base it on the actual topic discussed, not on greetings. "
                    "Return only the title text — no quotes, no punctuation, no explanation."
                ),
            },
            {"role": "user", "content": exchange},
        ],
        config=RunnableConfig(callbacks=[]),
    )

    title = get_message_text(response).strip()[:80]
    if not title:
        return {}

    # Dispatch custom event so the frontend sidebar updates immediately
    writer = get_stream_writer()
    writer({"type": "thread_title", "title": title})

    return {"thread_name": title}


def route_model_output(state: State) -> Literal["__end__", "consolidate_memories"]:
    """Route to background memory consolidation on a final answer, or end immediately
    when the model issued tool calls (already handled inside create_agent)."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage in output edges, but got {type(last_message).__name__}")
    if last_message.tool_calls:
        return "__end__"
    return "consolidate_memories"


def route_after_consolidation(state: State) -> Literal["__end__", "generate_thread_title"]:
    """Generate a thread title on the first complete exchange; otherwise finish."""
    if not state.thread_name:
        human_count = sum(1 for m in state.messages if isinstance(m, HumanMessage))
        if human_count == 1:
            return "generate_thread_title"
    return "__end__"


# Set of background tasks — prevents garbage collection of fire-and-forget tasks.
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]
_MAX_BACKGROUND_TASKS = 100


def _spawn_background_task(coro, label: str = "") -> None:
    """Create a fire-and-forget asyncio.Task with pool overflow protection."""
    task = asyncio.create_task(coro)
    if len(_background_tasks) >= _MAX_BACKGROUND_TASKS:
        logger.warning(
            "Background task pool full (%d tasks) — skipping %s",
            _MAX_BACKGROUND_TASKS,
            label,
        )
        task.cancel()
        return
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def consolidate_memories(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Post-response background work node.

    Dispatches fire-and-forget tasks so the graph reaches the next node
    (or ``__end__``) without delay:

    1. **Memory consolidation** — extracts durable facts into long-term store.
    2. **Session memory extraction** — updates session notes for compaction.

    Both run as ``asyncio.Task`` s to avoid blocking the response stream.
    Thread title generation remains a separate graph node because it needs
    to update graph state and send a custom event over the stream.
    """
    store = runtime.store
    user_id = runtime.context.user_id
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id", "")

    if not store or not user_id:
        return {}

    # ---- 1. Memory consolidation (existing logic) ----
    if not state.agent_wrote_memory:
        namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
        model = _build_runtime_model(runtime)
        messages = _normalize_messages_for_memory(list(state.messages))

        async def _run_consolidation() -> None:
            for attempt in range(2):
                try:
                    before = await store.asearch(namespace)
                    before_keys = {item.key for item in before}

                    stale_keys = [item.key for item in before if "kind" not in item.value]
                    if stale_keys:
                        logger.info(
                            "Purging %d old-format memory items for user=%s keys=%s",
                            len(stale_keys),
                            user_id,
                            stale_keys,
                        )
                        for key in stale_keys:
                            await store.adelete(namespace, key)
                        before_keys -= set(stale_keys)

                    manager = create_memory_store_manager(
                        model,
                        schemas=MEMORY_SCHEMAS,
                        namespace=namespace,
                        store=store,
                        enable_deletes=True,
                    )
                    await manager.ainvoke(
                        {"messages": messages},
                        config=RunnableConfig(callbacks=[]),
                    )

                    after = await store.asearch(namespace)
                    after_keys = {item.key for item in after}
                    created = after_keys - before_keys
                    deleted = before_keys - after_keys
                    updated = {
                        item.key
                        for item in after
                        if item.key in before_keys
                        and next((b.value for b in before if b.key == item.key), None) != item.value
                    }
                    logger.info(
                        "Memory consolidation complete — user=%s total=%d created=%d updated=%d deleted=%d created_keys=%s deleted_keys=%s",
                        user_id,
                        len(after_keys),
                        len(created),
                        len(updated),
                        len(deleted),
                        sorted(created),
                        sorted(deleted),
                    )

                    if len(after) > MAX_MEMORIES_PER_USER:
                        sorted_items = sorted(
                            after,
                            key=lambda item: (
                                getattr(item, "updated_at", None) or getattr(item, "created_at", None) or ""
                            ),
                        )
                        excess = len(after) - MAX_MEMORIES_PER_USER
                        pruned_keys = [item.key for item in sorted_items[:excess]]
                        for key in pruned_keys:
                            await store.adelete(namespace, key)
                        logger.info(
                            "Memory pruning — user=%s pruned=%d oldest keys=%s",
                            user_id,
                            excess,
                            pruned_keys,
                        )
                    break
                except Exception as exc:
                    exc_str = str(exc).lower()
                    is_connection_error = any(
                        marker in exc_str
                        for marker in (
                            "ssl error",
                            "eof detected",
                            "closed connection",
                            "connection reset",
                            "operationalerror",
                        )
                    )
                    if is_connection_error and attempt == 0:
                        logger.warning(
                            "Memory consolidation hit a transient connection error — retrying (attempt %d): %s",
                            attempt + 1,
                            str(exc)[:200],
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception("Background memory consolidation failed; continuing.")
                    break

        _spawn_background_task(_run_consolidation(), f"consolidation-{user_id}")
    else:
        logger.info(
            "Skipping background consolidation — agent wrote memories during hot path (user=%s)",
            user_id,
        )

    # ---- 2. Session memory extraction (background) ----
    if thread_id and should_extract_session_memory(
        list(state.messages),
        state.session_memory_token_count,
    ):
        _session_model_name = runtime.context.guardrail_model
        _session_messages = list(state.messages)

        async def _run_session_memory() -> None:
            try:
                _model = load_chat_model(_session_model_name)
                session_ns = get_session_namespace(user_id, thread_id)
                try:
                    existing = await store.asearch(session_ns, limit=1)
                    current_notes = (
                        existing[0].value.get("notes", SESSION_MEMORY_TEMPLATE) if existing else SESSION_MEMORY_TEMPLATE
                    )
                except Exception:
                    current_notes = SESSION_MEMORY_TEMPLATE

                updated_notes = await extract_session_memory(
                    _session_messages,
                    current_notes,
                    _model,
                )
                await store.aput(
                    session_ns,
                    "session_notes",
                    {"notes": updated_notes},
                )
                logger.debug("Session memory extraction complete — thread=%s", thread_id)
            except Exception:
                logger.debug("Background session memory extraction failed; continuing.", exc_info=True)

        _spawn_background_task(_run_session_memory(), f"session-mem-{thread_id}")

    return {}


async def update_session_memory(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Update session notes after each final AI response.

    Ported from Claude-code's ``SessionMemory`` post-sampling hook.

    Session notes capture in-progress work state that survives context
    compaction: what was being discussed, key decisions, errors encountered,
    and the worklog of steps taken.  They are injected into the system prompt
    via ``build_session_context_block``.

    The extraction is throttled by token growth since the last extraction
    to avoid excessive model calls.
    """
    store = runtime.store
    user_id = runtime.context.user_id
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    if not store or not user_id or not thread_id:
        return {}

    if not should_extract_session_memory(
        list(state.messages),
        state.session_memory_token_count,
    ):
        return {}

    # Use the guardrail model for extraction — cheap and fast
    model = load_chat_model(runtime.context.guardrail_model)

    # Load existing session notes from the store (or use template)
    session_ns = get_session_namespace(user_id, thread_id)
    try:
        existing = await store.asearch(session_ns, limit=1)
        current_notes = existing[0].value.get("notes", SESSION_MEMORY_TEMPLATE) if existing else SESSION_MEMORY_TEMPLATE
    except Exception:
        current_notes = SESSION_MEMORY_TEMPLATE

    try:
        updated_notes = await extract_session_memory(
            list(state.messages),
            current_notes,
            model,
        )

        # Persist to store
        await store.aput(
            session_ns,
            "session_notes",
            {"notes": updated_notes},
        )

        token_count = count_tokens_approximately(list(state.messages))
        return {
            "session_notes": updated_notes,
            "session_memory_token_count": token_count,
        }
    except Exception:
        logger.debug("Session memory extraction failed; continuing.", exc_info=True)
        return {}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the nodes
builder.add_node(screen_input)
builder.add_node(summarize)
builder.add_node(call_model)
builder.add_node(consolidate_memories)
builder.add_node(generate_thread_title)

# Entry: screen for injection → compress messages → call model
builder.add_edge("__start__", "screen_input")
builder.add_conditional_edges("screen_input", route_after_screening)
builder.add_edge("summarize", "call_model")
builder.add_edge("generate_thread_title", "__end__")

# call_model → consolidate_memories (final answer) or __end__ (tool call)
builder.add_conditional_edges("call_model", route_model_output)
# consolidate_memories dispatches memory + session extraction as background tasks,
# then routes to generate_thread_title (first turn) or __end__.
builder.add_conditional_edges("consolidate_memories", route_after_consolidation)

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
