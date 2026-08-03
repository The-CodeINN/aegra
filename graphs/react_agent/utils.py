"""Utility & helper functions."""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

_ENV_LOADED = False

# ---------------------------------------------------------------------------
# Model client cache
# ---------------------------------------------------------------------------
# Caches LLM client instances by (provider, model, enable_thinking, thinking_budget)
# so callers like generate_thread_title and screen_input don't build a new Bedrock
# or Anthropic client on every single turn.
_model_client_cache: dict[tuple[Any, ...], BaseChatModel] = {}
_model_client_cache_lock = asyncio.Lock()


def _find_env_file() -> Path | None:
    """Locate the nearest project .env file for local development."""
    search_roots = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()

    for root in search_roots:
        for candidate_root in [root, *root.parents]:
            if candidate_root in seen:
                continue
            seen.add(candidate_root)

            env_file = candidate_root / ".env"
            if env_file.is_file():
                return env_file

    return None


def _ensure_env_loaded() -> None:
    """Load local env vars so provider SDKs can resolve API keys."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_file = _find_env_file()
    if env_file:
        load_dotenv(env_file, override=False)

    _ENV_LOADED = True


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    fully_specified_name: str,
    enable_thinking: bool = False,
    thinking_budget: int = 10000,
) -> BaseChatModel:
    """Load a chat model from a fully specified name, with module-level caching.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        enable_thinking (bool): Whether to enable extended thinking for supported models.
        thinking_budget (int): Token budget for thinking (min 1024, max 128000).

    Returns a cached instance when called with the same arguments, avoiding
    redundant SDK client construction on every graph turn.
    """
    _ensure_env_loaded()

    cache_key = (fully_specified_name, enable_thinking, thinking_budget)
    # Fast path — no lock needed for a pure read on an already-populated cache
    if cache_key in _model_client_cache:
        return _model_client_cache[cache_key]

    provider, model = fully_specified_name.split("/", maxsplit=1)
    init_kwargs: dict[str, object] = {}

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        region = os.getenv("AWS_REGION_NAME") or os.getenv("AWS_DEFAULT_REGION", "eu-west-2")
        if enable_thinking:
            # Extended thinking requires temperature=1 on Bedrock
            client = ChatBedrockConverse(
                model=model,
                region_name=region,
                temperature=1,
                additional_model_request_fields={"thinking": {"type": "enabled", "budget_tokens": thinking_budget}},
            )
        else:
            client = ChatBedrockConverse(model=model, region_name=region, temperature=0)
    else:
        provider_api_key_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        api_key_env = provider_api_key_env.get(provider)
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                init_kwargs["api_key"] = api_key
        client = init_chat_model(model, model_provider=provider, **init_kwargs)

    # Populate cache — race condition is benign: worst case two callers build
    # the same client simultaneously and one is discarded.
    _model_client_cache[cache_key] = client
    return client
