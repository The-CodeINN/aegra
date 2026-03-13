"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from anthropic.types.beta import (
    BetaCodeExecutionTool20260120Param,
    BetaToolSearchToolBm25_20251119Param,
    BetaToolSearchToolRegex20251119Param,
)
from langchain.agents import create_agent
from langchain.tools import tool as lc_tool
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
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
    "search_course_content",
    "get_user_memory",
    "search_user_memories",
}


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

        return ChatAnthropic(**model_kwargs)

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


async def _invoke_integrated_agent(state: State, runtime: Runtime[Context], system_message: str) -> AIMessage:
    """Invoke a single integrated LangChain agent runtime for all providers."""
    agent = create_agent(
        model=_build_runtime_model(runtime),
        tools=_build_runtime_tools(runtime),
        middleware=_build_runtime_middleware(runtime),
        system_prompt=system_message,
    )

    result = await agent.ainvoke({"messages": list(state.messages)})
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg

    return AIMessage(content="I was unable to produce a response from the agent runtime.")


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

    response = await _invoke_integrated_agent(state, runtime, system_message)

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
builder.add_node(call_model)
builder.add_node(generate_thread_title)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")
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
