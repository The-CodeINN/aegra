"""Utility & helper functions."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

_ENV_LOADED = False


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
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        enable_thinking (bool): Whether to enable extended thinking for supported models.
        thinking_budget (int): Token budget for thinking (min 1024, max 128000).
    """
    _ensure_env_loaded()

    provider, model = fully_specified_name.split("/", maxsplit=1)
    init_kwargs: dict[str, object] = {}

    provider_api_key_env = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    api_key_env = provider_api_key_env.get(provider)
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            init_kwargs["api_key"] = api_key

    return init_chat_model(model, model_provider=provider, **init_kwargs)
