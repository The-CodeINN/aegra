# Golden conversation fixtures (spec Item 0)

Each `*.json` file here is one fixture consumed by
`tests/unit/react_agent/evals/replay.py` and asserted against in
`tests/unit/react_agent/evals/test_golden_conversations.py`.

## Fixture format

```json
{
  "name": "unique_fixture_name",
  "description": "what this fixture exercises and why",
  "context": { "...Context dataclass fields (graphs/react_agent/context.py)..." },
  "seed_memories": [{ "kind": "CareerGoal", "content": { "...schema fields..." } }],
  "messages": [{ "role": "user", "content": "..." }],
  "assertions": ["no_fabricated_linkedin_github", "no_system_prompt_leak"]
}
```

`assertions` must reference names registered in `_ASSERTIONS` in
`test_golden_conversations.py`. Add new assertion functions there as new
fixtures need them (e.g. once Item 2 lands, a `references_open_task`
assertion; once Item 4 lands, an `episodic_recall_present` assertion).

## Current fixtures are seed fixtures, not the real 20-30

The four fixtures currently in this directory are hand-authored to exercise
specific code paths (first-time roadmap trigger, returning-student memory
recall, a feedback/correction turn, a long single-sitting conversation for
Item 5's compaction-trigger regression test). They are **not** the 20-30 real
student conversations the spec calls for.

To add real conversations, run `scripts/export_golden_conversations.py`
against a running server with real thread history, then review/redact the
output before committing — conversations may contain PII and must be
scrubbed (names, emails, LinkedIn/GitHub URLs) before landing in this repo.
