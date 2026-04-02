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
from datetime import UTC, datetime
from typing import Any, Literal

from anthropic.types.beta import (
    BetaCodeExecutionTool20260120Param,
    BetaToolSearchToolBm25_20251119Param,
    BetaToolSearchToolRegex20251119Param,
)
from langchain.agents import create_agent
from langchain.tools import tool as lc_tool
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langmem import create_manage_memory_tool, create_memory_store_manager, create_search_memory_tool
from langmem.short_term import SummarizationNode

from react_agent.context import Context
from react_agent.guardrails import (
    INJECTION_BLOCKED_RESPONSE,
    LEAK_SAFE_RESPONSE,
    screen_input_for_injection,
    screen_output_for_leak,
)
from react_agent.memory import DEFAULT_MEMORY_NAMESPACE, MEMORY_SCHEMAS
from react_agent.message_utils import sanitize_messages_for_anthropic as _sanitize_messages
from react_agent.sanitized_anthropic import SanitizedChatAnthropic
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, load_chat_model

logger = logging.getLogger(__name__)

_NON_DEFERRED_TOOLS = {
    "get_student_profile",
    "get_student_onboarding",
    "brave_search",
}

_PROGRAMMATIC_SAFE_TOOLS = {
    "brave_search",
    "read_webpage",
    "get_student_profile",
    "get_student_onboarding",
    "get_student_ai_career_advisor_onboarding",
    "get_student_enrollment_overview",
    "get_course_structure",
    "get_course_materials",
    "get_course_progress",
    "get_student_attempts",
    "get_subscription_state",
    "get_portfolio_projects",
    "review_project_submission",
    "search_course_content",
    # LangMem long-term memory tools
    "manage_memory",
    "search_memory",
}

# Short-term memory token budgets (LangMem SummarizationNode)
_TRIM_MAX_TOKENS = 8000  # max tokens returned to call_model each turn
_SUMMARY_TRIGGER_TOKENS = 6000  # summarise when history exceeds this
_MAX_SUMMARY_TOKENS = 512  # budget for the generated summary itself

# Module-level cache: one SummarizationNode per model string to avoid
# rebuilding the LLM client on every graph turn.
_summarization_node_cache: dict[str, SummarizationNode] = {}


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
            normalized.append(msg)
    return normalized


# Long-term memory system-prompt hint (agent hot-path)
_MEMORY_INSTRUCTIONS = """

<memory_instructions>
You have long-term memory tools that persist facts across ALL conversations with this user:
- `search_memory`: use proactively at the start of each conversation and whenever prior user context may be relevant (goals, background, location, career targets, preferences).
- `manage_memory`: save any meaningful, durable information the user shares \u2014 career goals, target roles, location, skills, experience level, constraints, personal context, commitments.
- When the user corrects or updates something you stored, immediately update it with `manage_memory`.
- Always prefer recalled user context over generic responses.
</memory_instructions>"""


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


def _build_anthropic_tools(runtime: Runtime[Context]) -> list[Any]:
    """Build Anthropic-compatible tools with optional defer/caller metadata."""
    all_tools: list[Any] = list(TOOLS) + _build_langmem_tools(runtime)
    wrapped_tools: list[Any] = []

    for tool_fn in all_tools:
        tool_name = getattr(tool_fn, "name", getattr(tool_fn, "__name__", "tool"))
        extras: dict[str, Any] = {}

        if runtime.context.anthropic_tool_search_enabled and tool_name not in _NON_DEFERRED_TOOLS:
            extras["defer_loading"] = True

        if runtime.context.anthropic_programmatic_tool_calling_enabled and tool_name in _PROGRAMMATIC_SAFE_TOOLS:
            extras["allowed_callers"] = ["direct", "code_execution_20260120"]

        if extras:
            wrapped_tools.append(lc_tool(name_or_callable=tool_fn, extras=extras))
        else:
            wrapped_tools.append(tool_fn)

    if runtime.context.anthropic_tool_search_enabled:
        variant = (runtime.context.anthropic_tool_search_variant or "bm25").lower().strip()
        if variant == "regex":
            wrapped_tools.append(
                BetaToolSearchToolRegex20251119Param(
                    name="tool_search_tool_regex",
                    type="tool_search_tool_regex_20251119",
                )
            )
        else:
            wrapped_tools.append(
                BetaToolSearchToolBm25_20251119Param(
                    name="tool_search_tool_bm25",
                    type="tool_search_tool_bm25_20251119",
                )
            )

    if runtime.context.anthropic_programmatic_tool_calling_enabled:
        wrapped_tools.append(
            BetaCodeExecutionTool20260120Param(
                name="code_execution",
                type="code_execution_20260120",
            )
        )

    return wrapped_tools


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
    if _is_bedrock_model(runtime.context.model):
        # Bedrock Converse supports standard tool calling natively.
        return list(TOOLS) + _build_langmem_tools(runtime)
    if _is_anthropic_model(runtime.context.model):
        return _build_anthropic_tools(runtime)
    return list(TOOLS) + _build_langmem_tools(runtime)


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return get_message_text(message).strip()
    return ""


async def _invoke_integrated_agent(
    messages: list[AnyMessage], runtime: Runtime[Context], system_message: str
) -> AIMessage:
    """Invoke a single integrated LangChain agent runtime for all providers."""
    agent = create_agent(
        model=_build_runtime_model(runtime),
        tools=_build_runtime_tools(runtime),
        middleware=_build_runtime_middleware(runtime),
        system_prompt=system_message,
    )

    # Always normalise frontend multimodal content blocks (image/file/text-plain) into
    # the provider-specific format, and strip Anthropic beta incompatibilities from AIMessages.
    provider = "bedrock" if _is_bedrock_model(runtime.context.model) else "anthropic"
    messages = _sanitize_messages(messages, provider=provider)

    result = await agent.ainvoke({"messages": messages})
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg

    return AIMessage(content="I was unable to produce a response from the agent runtime.")


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
        return {"messages": normalized_messages} if needs_normalization else {}

    user_text = _latest_human_text(list(state.messages))
    if not user_text:
        return {"messages": normalized_messages} if needs_normalization else {}

    guardrail_model = load_chat_model(runtime.context.guardrail_model)
    is_injection = await screen_input_for_injection(user_text, guardrail_model)

    if not is_injection:
        return {"messages": normalized_messages} if needs_normalization else {}

    result: dict[str, Any] = {
        "messages": [AIMessage(content=INJECTION_BLOCKED_RESPONSE)],
        "guardrail_blocked": True,
    }
    return result


def route_after_screening(state: State) -> Literal["summarize", "__end__"]:
    """Route to ``__end__`` when an injection was blocked, otherwise to ``summarize``."""
    return "__end__" if state.guardrail_blocked else "summarize"


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
    """
    try:
        node = _get_summarization_node(runtime)
        return await node.ainvoke({"messages": list(state.messages), "context": dict(state.context or {})})
    except Exception:
        # Passthrough on failure so call_model always has something to work with.
        return {"summarized_messages": list(state.messages)}


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Call the LLM powering the agent.

    Uses ``state.summarized_messages`` (produced by the ``summarize`` node) as
    the message window so the model always operates within its token budget.

    Long-term memory is handled entirely by the agent's own ``manage_memory``
    and ``search_memory`` tools (LangMem hot path).  No manual extraction,
    keyword detection, or system-prompt injection is required here.
    """
    system_message = runtime.context.system_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    # Append the long-term memory tool usage hint.
    system_message += _MEMORY_INSTRUCTIONS

    # Use the token-bounded summarised messages from the previous node.
    # Fall back to the full message list on the very first turn (before
    # summarize has had a chance to produce output).
    prepared_messages = list(state.summarized_messages) or list(state.messages)

    response = await _invoke_integrated_agent(prepared_messages, runtime, system_message)

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
        response = AIMessage(content=LEAK_SAFE_RESPONSE, id=response.id)

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    return {"messages": [response]}


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
        runtime.context.model,
        enable_thinking=runtime.context.enable_thinking,
        thinking_budget=runtime.context.thinking_budget,
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
        ]
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


async def consolidate_memories(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Background memory extraction node.

    Runs ``create_memory_store_manager`` over the full thread after every final
    AI response to automatically extract, update, and consolidate durable facts
    into the user's long-term store — even when the agent didn't call
    ``manage_memory`` during the hot path.

    Uses the same ``MEMORY_SCHEMAS`` and namespace as the hot-path tools so
    all three writers share a consistent, structured store.

    The actual extraction work is dispatched as a fire-and-forget
    ``asyncio.Task`` so the memory-manager's internal LangGraph subgraph does
    NOT emit streaming events through the outer graph.  Without this, the
    frontend (which sets ``streamSubgraphs: true``) receives the
    extraction-LLM's ``<session_…>`` formatted prompt as a spurious ``values``
    event, causing a visible flicker at the end of the AI response stream.

    Errors are silently swallowed so a failure here never interrupts the
    conversation flow.
    """
    store = runtime.store
    user_id = runtime.context.user_id
    if not store or not user_id:
        return {}

    # Capture everything the background task needs BEFORE returning, since the
    # Runtime reference may not be valid after the graph node returns.
    namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
    model = _build_runtime_model(runtime)
    messages = _normalize_messages_for_memory(list(state.messages))

    async def _run() -> None:
        try:
            # Snapshot existing memories before consolidation so we can diff afterwards
            before = await store.asearch(namespace)
            before_keys = {item.key for item in before}

            # Purge old-format items that are missing the "kind" key required by the
            # current LangMem schema.  Older versions stored memories as plain dicts
            # ({"fact": "...", "category": "..."}) but the current version expects
            # {"kind": "TypeName", "content": {...}}.  Stale items cause a KeyError
            # inside extraction.py, so we delete them here and let the extractor
            # re-create them in the new format.
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
            await manager.ainvoke({"messages": messages})

            after = await store.asearch(namespace)
            after_keys = {item.key for item in after}
            created = after_keys - before_keys
            deleted = before_keys - after_keys
            updated = {
                item.key
                for item in after
                if item.key in before_keys and next((b.value for b in before if b.key == item.key), None) != item.value
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
        except Exception:
            logger.exception("Background memory consolidation failed; continuing.")

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

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
# consolidate_memories → generate_thread_title (first exchange) or __end__
builder.add_conditional_edges("consolidate_memories", route_after_consolidation)

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
