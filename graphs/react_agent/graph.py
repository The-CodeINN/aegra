"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import json
import re
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
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.guardrails import (
    INJECTION_BLOCKED_RESPONSE,
    LEAK_SAFE_RESPONSE,
    screen_input_for_injection,
    screen_output_for_leak,
)
from react_agent.memory import format_memory_value, get_user_memory_namespace, make_memory_key
from react_agent.message_utils import sanitize_messages_for_anthropic
from react_agent.sanitized_anthropic import SanitizedChatAnthropic
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import get_message_text, load_chat_model

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
    "get_user_memory",
    "search_user_memories",
}

_TRIM_MAX_TOKENS = 8000
_SUMMARY_TRIGGER_MESSAGES = 14
_SUMMARY_KEEP_RECENT = 8
_SUMMARY_MIN_NEW_MESSAGES = 4
_LONG_TERM_RECALL_LIMIT = 4
_LONG_TERM_STORE_LIMIT = 3


def _is_anthropic_model(model_name: str) -> bool:
    return model_name.split("/", maxsplit=1)[0].lower() == "anthropic"


def _extract_provider_model(model_name: str) -> str:
    return model_name.split("/", maxsplit=1)[1]


def _build_anthropic_tools(runtime: Runtime[Context]) -> list[Any]:
    """Build Anthropic-compatible tools with optional defer/caller metadata."""
    wrapped_tools: list[Any] = []

    for tool_fn in TOOLS:
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
    if _is_anthropic_model(runtime.context.model) and runtime.context.anthropic_prompt_caching_enabled:
        return [AnthropicPromptCachingMiddleware(ttl=runtime.context.anthropic_prompt_caching_ttl)]
    return []


def _build_runtime_tools(runtime: Runtime[Context]) -> list[Any]:
    if _is_anthropic_model(runtime.context.model):
        return _build_anthropic_tools(runtime)
    return list(TOOLS)


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return get_message_text(message).strip()
    return ""


def _message_role(message: AnyMessage) -> str:
    message_type = getattr(message, "type", message.__class__.__name__).lower()
    if message_type == "human":
        return "User"
    if message_type == "ai":
        return "Advisor"
    if message_type == "tool":
        return "Tool"
    return message_type.title()


def _clean_json_block(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"(\[.*\])", stripped, re.DOTALL)
    return match.group(1) if match else stripped


def _should_extract_memories(user_text: str) -> bool:
    normalized = user_text.lower()
    if len(normalized.strip()) < 24:
        return False
    triggers = (
        "i am ",
        "i'm ",
        "my goal",
        "i want",
        "i need",
        "i prefer",
        "i like",
        "i work",
        "i worked",
        "my background",
        "my experience",
        "remember",
        "i have",
    )
    return any(trigger in normalized for trigger in triggers)


async def _recall_long_term_memories(state: State, runtime: Runtime[Context]) -> list[str]:
    store = runtime.store
    user_id = runtime.context.user_id
    query = _latest_human_text(list(state.messages))

    if not store or not user_id or not query:
        return []

    namespace = get_user_memory_namespace(user_id)

    try:
        try:
            results = await store.asearch(namespace, query=query, limit=_LONG_TERM_RECALL_LIMIT)
        except Exception:
            results = await store.asearch(namespace, limit=_LONG_TERM_RECALL_LIMIT)
    except Exception:
        return []

    recalled: list[str] = []
    for item in results:
        if isinstance(item.value, dict):
            rendered = format_memory_value(item.value)
            if rendered:
                recalled.append(rendered)
    return recalled[:_LONG_TERM_RECALL_LIMIT]


async def _maybe_refresh_summary(state: State, runtime: Runtime[Context]) -> tuple[str, int]:
    messages = list(state.messages)
    existing_summary = state.conversation_summary.strip()
    already_summarized = state.summary_message_count

    if len(messages) < _SUMMARY_TRIGGER_MESSAGES:
        return existing_summary, already_summarized

    candidate_count = max(0, len(messages) - _SUMMARY_KEEP_RECENT)
    if candidate_count <= already_summarized:
        return existing_summary, already_summarized
    if candidate_count - already_summarized < _SUMMARY_MIN_NEW_MESSAGES:
        return existing_summary, already_summarized

    transcript_lines = []
    for message in messages[:candidate_count]:
        content = get_message_text(message).strip()
        if content:
            transcript_lines.append(f"{_message_role(message)}: {content}")

    if not transcript_lines:
        return existing_summary, already_summarized

    model = _build_runtime_model(runtime)
    summary_prompt = (
        "You maintain a running short-term memory summary for a career mentor. "
        "Summarize only durable conversational context: the student's goals, background, "
        "constraints, current projects, blockers, and important commitments. "
        "Keep it under 220 words and avoid fluff.\n\n"
    )
    if existing_summary:
        summary_prompt += f"Existing summary:\n{existing_summary}\n\n"
    summary_prompt += "Conversation segment to summarize:\n" + "\n".join(transcript_lines)

    response = await model.ainvoke(
        [
            {"role": "system", "content": "Return only the updated summary text."},
            {"role": "user", "content": summary_prompt},
        ]
    )
    new_summary = get_message_text(response).strip()
    return (new_summary or existing_summary), candidate_count


def _augment_system_prompt(system_message: str, conversation_summary: str, long_term_memories: list[str]) -> str:
    blocks: list[str] = []

    if conversation_summary:
        blocks.append(
            f"<short_term_memory>\nRunning summary of this thread:\n{conversation_summary}\n</short_term_memory>"
        )

    if long_term_memories:
        rendered_memories = "\n".join(f"- {memory}" for memory in long_term_memories)
        blocks.append(
            "<long_term_memory>\n"
            "Recalled user memories from prior conversations. Use them when relevant, but if the user "
            "corrects or contradicts any memory, prefer the new information immediately.\n"
            f"{rendered_memories}\n"
            "</long_term_memory>"
        )

    if not blocks:
        return system_message
    return system_message + "\n\n" + "\n\n".join(blocks)


def _prepare_messages_for_model(state: State, summary_available: bool) -> list[AnyMessage]:
    messages = list(state.messages)
    if summary_available and len(messages) > _SUMMARY_KEEP_RECENT:
        messages = messages[-_SUMMARY_KEEP_RECENT:]

    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=_TRIM_MAX_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False,
        allow_partial=False,
    )
    return list(trimmed)


async def _persist_long_term_memories(state: State, runtime: Runtime[Context], response: AIMessage) -> None:
    store = runtime.store
    user_id = runtime.context.user_id
    user_text = _latest_human_text(list(state.messages))

    if not store or not user_id or not _should_extract_memories(user_text):
        return

    model = _build_runtime_model(runtime)
    extraction_prompt = (
        "Extract durable long-term memories from the user's message for a career mentor. "
        "Only include stable facts, preferences, goals, background, constraints, or project history that "
        "will matter in future conversations. Ignore transient requests and anything not explicitly stated. "
        "Return strict JSON as an array of objects with keys 'category' and 'text'. If nothing should be "
        "stored, return [].\n\n"
        f"User message:\n{user_text}\n\n"
        f"Assistant reply for context:\n{get_message_text(response).strip()}"
    )

    raw_result = await model.ainvoke(
        [
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": extraction_prompt},
        ]
    )
    raw_text = _clean_json_block(get_message_text(raw_result))

    try:
        candidates = json.loads(raw_text)
    except json.JSONDecodeError:
        return

    if not isinstance(candidates, list):
        return

    namespace = get_user_memory_namespace(user_id)
    stored = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        category = str(candidate.get("category", "note")).strip().lower() or "note"
        text = str(candidate.get("text", "")).strip()
        if not text:
            continue
        key = make_memory_key(category, text)
        await store.aput(
            namespace,
            key,
            {
                "category": category,
                "text": text,
                "source": "auto",
                "captured_at": datetime.now(tz=UTC).isoformat(),
            },
        )
        stored += 1
        if stored >= _LONG_TERM_STORE_LIMIT:
            break


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

    # Sanitize messages for Anthropic compatibility (removes problematic fields like index from tool_search_tool_result)
    sanitized_messages = sanitize_messages_for_anthropic(messages)

    result = await agent.ainvoke({"messages": sanitized_messages})
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg

    return AIMessage(content="I was unable to produce a response from the agent runtime.")


async def screen_input(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Pre-screen the latest user message for prompt-injection / jailbreak attempts.

    If guardrails are disabled or no injection is detected, returns an empty
    dict so the graph proceeds to ``call_model`` unchanged.

    If an injection is detected, adds a safe refusal ``AIMessage`` and sets
    ``guardrail_blocked=True`` so the graph short-circuits to ``__end__``.
    """
    if not runtime.context.guardrails_enabled:
        return {}

    user_text = _latest_human_text(list(state.messages))
    if not user_text:
        return {}

    guardrail_model = load_chat_model(runtime.context.guardrail_model)
    is_injection = await screen_input_for_injection(user_text, guardrail_model)

    if not is_injection:
        return {}

    return {
        "messages": [AIMessage(content=INJECTION_BLOCKED_RESPONSE)],
        "guardrail_blocked": True,
    }


def route_after_screening(state: State) -> Literal["call_model", "__end__"]:
    """Route to ``__end__`` when an injection was blocked, otherwise to ``call_model``."""
    return "__end__" if state.guardrail_blocked else "call_model"


# Define the function that calls the model


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, list[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
    conversation_summary, summary_message_count = await _maybe_refresh_summary(state, runtime)
    recalled_memories = await _recall_long_term_memories(state, runtime)
    system_message = _augment_system_prompt(system_message, conversation_summary, recalled_memories)
    prepared_messages = _prepare_messages_for_model(state, summary_available=bool(conversation_summary))

    response = await _invoke_integrated_agent(prepared_messages, runtime, system_message)

    # Output screening: replace the response if it leaks system-prompt content.
    if runtime.context.guardrails_enabled and screen_output_for_leak(get_message_text(response)):
        response = AIMessage(content=LEAK_SAFE_RESPONSE, id=response.id)

    await _persist_long_term_memories(state, runtime, response)

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

    # Return the model's response as a list to be added to existing messages
    updates: dict[str, Any] = {"messages": [response]}
    if conversation_summary != state.conversation_summary:
        updates["conversation_summary"] = conversation_summary
    if summary_message_count != state.summary_message_count:
        updates["summary_message_count"] = summary_message_count
    return updates


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


def route_model_output(state: State) -> Literal["__end__", "generate_thread_title"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage in output edges, but got {type(last_message).__name__}")
    # Tool calls are handled inside create_agent; no external ToolNode loop.
    if last_message.tool_calls:
        return "__end__"
    # Generate a title once — only on the first complete exchange
    if not state.thread_name:
        human_count = sum(1 for m in state.messages if isinstance(m, HumanMessage))
        if human_count == 1:
            return "generate_thread_title"
    return "__end__"


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the nodes
builder.add_node(screen_input)
builder.add_node(call_model)
builder.add_node(generate_thread_title)

# Entry: screen for injection before reaching the main model
builder.add_edge("__start__", "screen_input")
builder.add_conditional_edges("screen_input", route_after_screening)
builder.add_edge("generate_thread_title", "__end__")


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")
