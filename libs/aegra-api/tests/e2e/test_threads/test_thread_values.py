"""E2E: Thread responses carry the latest ``values`` and ``interrupts`` (issue #648)."""

import json

import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_get_and_search_return_latest_values_after_run() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    assert thread["values"] is None
    assert thread["interrupts"] == {}

    await client.runs.wait(
        thread_id,
        assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"steps": 1, "delay": 0})}]},
    )

    fetched = await client.threads.get(thread_id)
    elog("threads.get after run", fetched)
    state = await client.threads.get_state(thread_id)
    assert fetched["values"] == state["values"]
    assert fetched["values"]["messages"][-1]["type"] == "ai"
    assert fetched["interrupts"] == {}

    [searched] = [t for t in await client.threads.search(limit=100) if t["thread_id"] == thread_id]
    assert searched["values"] == state["values"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_thread_interrupts_are_keyed_by_task_and_clear_on_resume() -> None:
    client = get_e2e_client()
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    await client.runs.wait(thread_id, "subgraph_hitl_agent", input={"foo": "Test value."})

    interrupted = await client.threads.get(thread_id)
    elog("threads.get while interrupted", interrupted)
    assert interrupted["status"] == "interrupted"
    [task_interrupts] = interrupted["interrupts"].values()
    assert [i["value"] for i in task_interrupts] == ["Provide value:"]
    assert task_interrupts[0]["id"]

    await client.runs.wait(thread_id, "subgraph_hitl_agent", command={"resume": " resumed"})

    resumed = await client.threads.get(thread_id)
    elog("threads.get after resume", resumed)
    assert resumed["status"] == "idle"
    assert resumed["interrupts"] == {}
    assert resumed["values"]["foo"].endswith(" resumed")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_update_state_refreshes_thread_values() -> None:
    client = get_e2e_client()
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    await client.runs.wait(
        thread_id,
        assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"steps": 1, "delay": 0})}]},
    )

    await client.threads.update_state(thread_id, {"step_count": 42}, as_node="respond")

    fetched = await client.threads.get(thread_id)
    elog("threads.get after update_state", fetched)
    assert fetched["values"]["step_count"] == 42
