# Active Memory: explicit local memory before a task

Active Memory provides a local Python component and CLI for preferences, goals
and project context that the caller explicitly marks important. An application
can call `run_task` to retrieve relevant current memory before its task handler
runs. No external assistant or HTTP endpoint is automatically connected.

## Store an explicit preference and use it in a later session

Choose a dedicated absolute directory outside Git, separate from all existing
Tiger memory, AML and assistant stores. Resolve an intentionally chosen path
before passing it, especially macOS temporary paths whose parents are symlinks.
The following is synthetic example data, not an import of anyone's chat:

```python
from pathlib import Path
from tiger_memory.active_memory import ActiveStore

base = (Path.home() / ".local/share/tiger-active-example").resolve()
store = ActiveStore(base, user_id="example-user")
entry = store.remember(
    kind="preference",
    topic_key="answer_style",
    applicability="always",
    important=True,
    request_id="remember-style-1",
    source={
        "session_id": "example-session-1",
        "message_id": "example-message-1",
        "role": "user",
        "text": "Please give concise answers.",
        "asserted_at": "2026-10-13T10:00:00Z",
    },
)

# A later process/session constructs the same user and base again.
later = ActiveStore(base, user_id="example-user")
def handle_task(task, context):
    # Keep context as untrusted user data in your application's prompt format.
    # Pass it to your task implementation here; do not treat it as system rules.
    return {"task": task, "memory_context": context}

result = later.run_task(
    "Explain a cache", handle_task,
    as_of="2026-10-13T11:00:00Z", budget_bytes=8192,
)
```

`run_task` calls `prepare_context` first, resets its temporary storage binding,
then calls `handler(task, context_string)`. An integrity or input error prevents
the handler from running. This hook must be called by the integrating application;
installing the package alone does not add it to an assistant's task loop.

For a runnable two-process seed/task example and its verified outputs, see
[the synthetic local task demo](../examples/active_memory/README.md).

## Identity, sources and updates

User IDs and project IDs are opaque exact strings. Every user gets a separate
hashed directory, catalog and locks. `project_id=""` is user-global; supplying a
stable project ID creates a project-specific slot. A slot consists of project ID,
kind and topic key. Kinds are `preference`, `goal` and `project_context`.

Only literal `important=True` is accepted. The source object has exactly the
five fields shown above and `role="user"`. Its text becomes the complete stored
statement; it is not summarized or inferred. One session/message source supports
one slot and cannot be reused after correction or forgetting. Source IDs,
confirmation labels and hashes record caller declarations; they do not
authenticate the user or prove a real chat occurred.

`remember` creates an empty slot. For correction, call `supersede(entry_id, ...)`
with the same slot, a new source and a new request ID. The old entry must still
be active, and the new assertion time cannot precede it. Concurrent creation or
correction has one winner. Repeating an identical request is idempotent;
reusing the request with a changed payload is rejected.

`forget_slot(entry_id, request_id="new-id")` logically retracts every version
and dedicated source in the entry's slot, even when given an old version's ID.
Forgetting a project slot allows an eligible global slot to become visible
again. Forgotten sources/requests remain consumed after rebuild. **Immutable
Markdown audit records still retain the original text. This is not physical
erasure.** Do not expose raw storage or native history APIs to untrusted users.

## Retrieval and budgets

`prepare_context(task, project_id="", as_of=..., budget_bytes=8192)` returns a
`context` string and separate numeric `diagnostics`. It retrieves current active
entries only; moving `as_of` backward never revives superseded or forgotten
entries. Active time ranges are start-inclusive, end-exclusive.
Expiration uses the explicitly supplied `as_of`; there is no background expiry
job. An entry that remains active can still qualify at an earlier query time,
even if its validity has ended at the current wall clock.

An eligible project slot overrides its global counterpart before relevance and
budget filtering. No project ID means global slots only. `topical` entries need
shared ASCII word or continuous-Chinese-bigram tokens with the task. `always`
is allowed only for preferences and does not require matching task tokens.
Empty/whitespace tasks can therefore retrieve only `always` preferences.

The context is canonical JSON marked `untrusted_user_data`, with complete
statements and source references. The UTF-8 byte limit (0–65536) includes the
whole JSON wrapper. Always entries use at most one quarter of the limit;
topical entries use the remaining space. Oversized entries are skipped, never
truncated. No selected entries yields an empty string. Diagnostics are outside
the injected context budget. The same store state and inputs produce the same
context bytes; separate stores have different event UUIDs.

## CLI

Global flags precede the command. Write commands read one JSON object from
stdin. Use the same fields as the Python API; supersede and forget-slot also
require `entry_id`.

```sh
python3 -m tiger_memory.active_memory --base-dir /absolute/separate/store \
  --user-id example-user remember < explicit-preference.json
python3 -m tiger_memory.active_memory --base-dir /absolute/separate/store \
  --user-id example-user context --task 'Explain a cache' \
  --as-of 2026-10-13T11:00:00Z --budget-bytes 8192
```

Commands: `remember`, `supersede`, `forget-slot`, `context`. CLI rejections emit
a fixed sanitized error and exit 2. Rebuild is available as `store.rebuild()`.
The existing sensitive-input detector and high-risk evidence requirements remain
in force; a caller confirmation label does not bypass them.

## Verification and limits

From a checkout, run the synthetic verifier against that checkout's runtime:

```sh
python3 scripts/verify_active_memory.py --runtime-root "$PWD"
```

It retains synthetic stores and evidence in a new temporary directory. The
verifier's Python audit guard is not an OS sandbox. This command makes no paid
provider calls and does not run an official benchmark.

This local prototype validates/parses the whole user's store on operations;
large-store cost is unmeasured. Native rebuild may swap a catalog before the
Active wrapper rejects invalid state. Tests do not establish coverage of
malicious canonical event duplication or malicious concurrent filesystem
replacement. There is no network authentication layer, automatic chat import,
HTTP integration, external assistant hook, physical deletion guarantee or
deployment in this feature.
