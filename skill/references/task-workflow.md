# Project memory workflow

Use the installed skill's `scripts/memory_context.py --cwd <project> --query
<focused-question>` to retrieve relevant history. The current source also
provides `scripts/tiger_memory.py context` with JSON, passage and scope options. Inspect
`--help` for supported arguments; other installed versions may expose different
commands.

Treat retrieved records as sourced hints. Check current files, Git and test
results when the answer depends on current implementation or release state.
An absent search result is not proof that an event never occurred.

When code evidence is needed, use an available CodeGraph MCP directly, or use
`tiger_memory.py context --with-code` against a previously bound local index. Avoid
duplicate retrieval. Index creation or refresh requires authorization.

Write only durable information within the user's authorization. Preserve its
source, project scope, conditions, dates and negation. Agent suggestions are not
user decisions; local tests are not deployment or user acceptance. Use
`candidate`, `candidate-review` and `consolidate` for structured changes.
High-risk facts need evidence bound to the exact claim. Do not invent source
IDs or confirmation receipts.

Use `history`, `supersede` and `forget` for lifecycle changes. Do not edit the
derived SQLite index directly. Never store credentials or unnecessary personal
data. Follow the user's permission boundaries for both reads and writes.
