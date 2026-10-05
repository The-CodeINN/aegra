"""Pin every v1 request body ``langgraph_sdk`` sends to the keys Aegra declares.

Each SDK method is called with a value for every parameter it accepts, so a new
SDK parameter fails here until ``SDK_ARGS`` has a value for it and the server
either declares the key or it is listed in ``KNOWN_GAPS`` (tracked in #503).
"""

import inspect
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FunctionType
from typing import Any

import httpx
import pytest
from langgraph_sdk.client import (
    AssistantsClient,
    CronClient,
    HttpClient,
    RunsClient,
    StoreClient,
    ThreadsClient,
)
from pydantic import BaseModel

from aegra_api.models.assistants import AssistantCreate, AssistantSearchRequest, AssistantUpdate
from aegra_api.models.crons import CronCountRequest, CronCreate, CronSearchRequest, CronUpdate
from aegra_api.models.runs import RunCreate, RunsCancel
from aegra_api.models.store import (
    StoreDeleteRequest,
    StoreListNamespacesRequest,
    StorePutRequest,
    StoreSearchRequest,
)
from aegra_api.models.threads import (
    ThreadCheckpointPostRequest,
    ThreadCreate,
    ThreadHistoryRequest,
    ThreadSearchRequest,
    ThreadStateUpdate,
    ThreadUpdate,
)

SdkClient = AssistantsClient | ThreadsClient | RunsClient | CronClient | StoreClient
SDK_CLIENTS: tuple[type[SdkClient], ...] = (AssistantsClient, ThreadsClient, RunsClient, CronClient, StoreClient)

# One value per SDK parameter name; a parameter missing here fails the test.
SDK_ARGS: dict[str, Any] = {
    "action": "interrupt",
    "after_seconds": 5,
    "as_node": "node",
    "assistant_id": "asst-1",
    "before": "cp-0",
    "cancel_on_disconnect": True,
    "checkpoint": {"checkpoint_id": "cp-1"},
    "checkpoint_during": True,
    "checkpoint_id": "cp-1",
    "command": {"resume": "yes"},
    "config": {"recursion_limit": 7},
    "context": {"model": "m"},
    "cron_id": "cron-1",
    "delete_threads": True,
    "description": "described",
    "durability": "sync",
    "enabled": True,
    "end_time": datetime(2030, 1, 1, tzinfo=UTC),
    "extract": {"title": "values.title"},
    "feedback_keys": ["score"],
    "filter": {"kind": "note"},
    "graph_id": "graph",
    "ids": ["thread-1"],
    "if_exists": "do_nothing",
    "if_not_exists": "create",
    "include": ["values"],
    "index": ["text"],
    "input": {"messages": []},
    "interrupt_after": ["node"],
    "interrupt_before": ["node"],
    "key": "item-1",
    "langsmith_tracing": {"project_name": "p"},
    "last_event_id": "evt-1",
    "limit": 5,
    "max_depth": 2,
    "metadata": {"tenant": "t"},
    "multitask_strategy": "enqueue",
    "name": "named",
    "namespace": ["users", "u1"],
    "namespace_prefix": ["users"],
    "offset": 1,
    "on_completion": "keep",
    "on_disconnect": "cancel",
    "on_run_completed": "keep",
    "payloads": [{"assistant_id": "asst-1"}],
    "prefix": ["users"],
    "query": "q",
    "recurse": True,
    "refresh_ttl": True,
    "return_minimal": True,
    "run_ids": ["run-1"],
    "run_id": "run-1",
    "schedule": "0 * * * *",
    "select": ["thread_id"],
    "sort_by": "created_at",
    "sort_order": "asc",
    "status": "pending",
    "stream_mode": ["values"],
    "stream_resumable": True,
    "stream_subgraphs": True,
    "strategy": "delete",
    "subgraphs": True,
    "suffix": ["notes"],
    "supersteps": [{"updates": [{"values": {}, "as_node": "node"}]}],
    "thread_id": "thread-1",
    "thread_ids": ["thread-1"],
    "timezone": "UTC",
    "ttl": {"strategy": "delete", "ttl": 60},
    "value": {"text": "hi"},
    "values": {"title": "t"},
    "version": 1,
    "wait": True,
    "webhook": "https://example.com/hook",
    "xray": True,
}

# Client-side only: transport options, callbacks and response parsing; never in the body.
CLIENT_ONLY_PARAMS: frozenset[str] = frozenset(
    {"headers", "params", "on_run_created", "raise_error", "response_format"}
)


@dataclass(frozen=True)
class BodyCase:
    client: type[SdkClient]
    method: str
    declared: frozenset[str]


def _fields(model: type[BaseModel]) -> frozenset[str]:
    return frozenset(model.model_fields)


BODY_CASES: tuple[BodyCase, ...] = (
    BodyCase(AssistantsClient, "create", _fields(AssistantCreate)),
    BodyCase(AssistantsClient, "update", _fields(AssistantUpdate)),
    BodyCase(AssistantsClient, "search", _fields(AssistantSearchRequest)),
    BodyCase(AssistantsClient, "count", _fields(AssistantSearchRequest)),
    BodyCase(AssistantsClient, "set_latest", frozenset({"version"})),
    BodyCase(AssistantsClient, "get_versions", frozenset()),
    BodyCase(ThreadsClient, "create", _fields(ThreadCreate)),
    BodyCase(ThreadsClient, "update", _fields(ThreadUpdate)),
    BodyCase(ThreadsClient, "search", _fields(ThreadSearchRequest)),
    BodyCase(ThreadsClient, "get_history", _fields(ThreadHistoryRequest)),
    BodyCase(ThreadsClient, "get_state", _fields(ThreadCheckpointPostRequest)),
    BodyCase(ThreadsClient, "update_state", _fields(ThreadStateUpdate)),
    BodyCase(ThreadsClient, "prune", frozenset()),
    BodyCase(RunsClient, "create", _fields(RunCreate)),
    BodyCase(RunsClient, "stream", _fields(RunCreate)),
    BodyCase(RunsClient, "wait", _fields(RunCreate)),
    BodyCase(RunsClient, "cancel_many", _fields(RunsCancel)),
    BodyCase(CronClient, "create", _fields(CronCreate)),
    BodyCase(CronClient, "create_for_thread", _fields(CronCreate)),
    BodyCase(CronClient, "update", _fields(CronUpdate)),
    BodyCase(CronClient, "search", _fields(CronSearchRequest)),
    BodyCase(CronClient, "count", _fields(CronCountRequest)),
    BodyCase(StoreClient, "put_item", _fields(StorePutRequest)),
    BodyCase(StoreClient, "search_items", _fields(StoreSearchRequest)),
    BodyCase(StoreClient, "list_namespaces", _fields(StoreListNamespacesRequest)),
    BodyCase(StoreClient, "delete_item", _fields(StoreDeleteRequest)),
)

# SDK methods that send no JSON body, or whose route Aegra does not serve yet.
NO_BODY_METHODS: frozenset[tuple[str, str]] = frozenset(
    {
        ("AssistantsClient", "delete"),
        ("AssistantsClient", "get"),
        ("AssistantsClient", "get_graph"),
        ("AssistantsClient", "get_schemas"),
        ("AssistantsClient", "get_subgraphs"),
        ("ThreadsClient", "delete"),
        ("ThreadsClient", "get"),
        ("ThreadsClient", "join_stream"),
        ("RunsClient", "cancel"),
        ("RunsClient", "delete"),
        ("RunsClient", "get"),
        ("RunsClient", "join"),
        ("RunsClient", "join_stream"),
        ("RunsClient", "list"),
        ("CronClient", "delete"),
        ("StoreClient", "get_item"),
    }
)
UNROUTED_METHODS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ThreadsClient", "copy"),
        ("ThreadsClient", "count"),
        ("RunsClient", "create_batch"),
    }
)
# Agent Protocol v2 websocket stream, pinned by test_event_streaming/test_spec_params.py.
V2_METHODS: frozenset[tuple[str, str]] = frozenset({("ThreadsClient", "stream")})

# Body keys the SDK sends that the server currently drops. This list may only shrink.
KNOWN_GAPS: dict[tuple[str, str], frozenset[str]] = {
    ("AssistantsClient", "search"): frozenset({"select"}),
    ("AssistantsClient", "get_versions"): frozenset({"limit", "offset", "metadata"}),
    ("ThreadsClient", "create"): frozenset({"supersteps"}),
    ("ThreadsClient", "update"): frozenset({"ttl"}),
    ("ThreadsClient", "search"): frozenset({"ids", "values", "select", "extract"}),
    ("ThreadsClient", "prune"): frozenset({"thread_ids"}),
    ("RunsClient", "create"): frozenset(
        {
            "after_seconds",
            "checkpoint_during",
            "durability",
            "if_not_exists",
            "langsmith_tracer",
            "stream_resumable",
            "webhook",
        }
    ),
    ("RunsClient", "stream"): frozenset(
        {
            "after_seconds",
            "checkpoint_during",
            "durability",
            "feedback_keys",
            "if_not_exists",
            "langsmith_tracer",
            "stream_resumable",
            "webhook",
        }
    ),
    ("RunsClient", "wait"): frozenset(
        {"after_seconds", "checkpoint_during", "durability", "if_not_exists", "langsmith_tracer", "webhook"}
    ),
    ("CronClient", "create"): frozenset({"checkpoint_during", "durability", "stream_resumable"}),
    ("CronClient", "create_for_thread"): frozenset({"checkpoint_during", "durability", "stream_resumable"}),
    ("CronClient", "update"): frozenset({"durability", "stream_resumable"}),
    ("CronClient", "search"): frozenset({"select"}),
    ("StoreClient", "put_item"): frozenset({"index", "ttl"}),
    ("StoreClient", "search_items"): frozenset({"refresh_ttl"}),
}


class _BodyCaptured(Exception):
    """Raised by the mock transport so no SDK response parsing runs."""


@dataclass
class CapturedRequest:
    method: str
    path: str
    body: Any


def _key(case: BodyCase) -> tuple[str, str]:
    return case.client.__name__, case.method


def _public_methods(client: type[SdkClient]) -> list[str]:
    return [name for name, _ in inspect.getmembers(client, inspect.isfunction) if not name.startswith("_")]


def _sdk_call_args(func: FunctionType) -> tuple[list[Any], dict[str, Any]]:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    params = list(inspect.signature(func).parameters.values())[1:]
    for param in params:
        if param.name in CLIENT_ONLY_PARAMS or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.name not in SDK_ARGS:
            pytest.fail(f"{func.__qualname__} gained parameter {param.name!r}: add a value to SDK_ARGS")
        if param.kind is param.POSITIONAL_ONLY:
            args.append(SDK_ARGS[param.name])
        else:
            kwargs[param.name] = SDK_ARGS[param.name]
    return args, kwargs


async def _capture(client: type[SdkClient], method: str) -> CapturedRequest:
    captured: list[CapturedRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        captured.append(CapturedRequest(request.method, request.url.path, body))
        raise _BodyCaptured

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://aegra.test", transport=transport) as raw:
        sdk = client(HttpClient(raw))
        func = getattr(client, method)
        args, kwargs = _sdk_call_args(func)
        result = getattr(sdk, method)(*args, **kwargs)
        with pytest.raises(_BodyCaptured):
            if inspect.isawaitable(result):
                result = await result
            async for _ in result:
                pass

    assert len(captured) == 1
    return captured[0]


# ``checkpoint_during`` is deprecated in the SDK but still sent, so it stays covered.
pytestmark = pytest.mark.filterwarnings("ignore:`checkpoint_during` is deprecated:DeprecationWarning")


@pytest.mark.parametrize("client", SDK_CLIENTS, ids=lambda c: c.__name__)
def test_every_sdk_method_is_classified(client: type[SdkClient]) -> None:
    classified = {_key(c) for c in BODY_CASES} | NO_BODY_METHODS | UNROUTED_METHODS | V2_METHODS

    unclassified = {(client.__name__, m) for m in _public_methods(client)} - classified

    assert not unclassified, f"new SDK methods need a BodyCase or a classification: {sorted(unclassified)}"


@pytest.mark.parametrize("case", BODY_CASES, ids=lambda c: f"{c.client.__name__}.{c.method}")
async def test_sdk_body_keys_are_declared_or_known_gaps(case: BodyCase) -> None:
    request = await _capture(case.client, case.method)

    assert isinstance(request.body, dict), f"{request.method} {request.path} sent no JSON object"
    undeclared = set(request.body) - case.declared
    known = KNOWN_GAPS.get(_key(case), frozenset())
    assert undeclared <= known, f"{request.method} {request.path} drops SDK keys: {sorted(undeclared - known)}"


@pytest.mark.parametrize("case", BODY_CASES, ids=lambda c: f"{c.client.__name__}.{c.method}")
async def test_known_gaps_are_not_stale(case: BodyCase) -> None:
    """A closed gap must leave KNOWN_GAPS so it cannot silently reopen."""
    request = await _capture(case.client, case.method)

    known = KNOWN_GAPS.get(_key(case), frozenset())
    still_dropped = set(request.body) - case.declared

    assert known <= still_dropped, f"remove fixed keys from KNOWN_GAPS: {sorted(known - still_dropped)}"


def test_known_gaps_name_real_body_cases() -> None:
    assert set(KNOWN_GAPS) <= {_key(c) for c in BODY_CASES}


@pytest.mark.parametrize("method", sorted(NO_BODY_METHODS), ids=lambda m: ".".join(m))
async def test_no_body_method_sends_no_json(method: tuple[str, str]) -> None:
    client_name, name = method
    client = next(c for c in SDK_CLIENTS if c.__name__ == client_name)

    request = await _capture(client, name)

    assert request.body is None, f"{client_name}.{name} now sends a body: add a BodyCase"
