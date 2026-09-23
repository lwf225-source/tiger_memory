# PLM v2 Runbook

Runtime entrypoint:

```bash
PLM="$HOME/.codex/skills/project-long-memory/scripts/plm.py"
python3 "$PLM" config
python3 "$PLM" doctor
python3 "$PLM" usage --days 7
```

## Recall and write

```bash
python3 "$PLM" context --cwd "$PWD" --query "current release status" --explain
python3 "$PLM" write --cwd "$PWD" --title "Decision" --type episode \
  --tags "decision" --content "Sourced durable context"
```

Use `--scope project` by default. Agent, session, shared, and global records
must be queried with the matching scope and optional `--scope-id`.

## Facts and candidates

```bash
python3 "$PLM" history --cwd "$PWD" --fact-key "release.status"
python3 "$PLM" supersede --cwd "$PWD" --fact-id '<id>' --value 'new value'
python3 "$PLM" candidate --cwd "$PWD" --payload '<structured-json>'
python3 "$PLM" consolidate
```

High-risk status candidates without user confirmation or verifiable tool
evidence remain `needs_confirmation`.

`usage` reports invocation counts, result counts, success and latency. It stores
only a SHA-256 query hash, never the query text or retrieved memory body.

## Repair and recovery

```bash
python3 "$PLM" backup
python3 "$PLM" rebuild-index
python3 "$PLM" verify
python3 "$PLM" shadow-report
```

`rebuild-index` builds and validates a temporary SQLite catalog, preserves the
old catalog as a timestamped backup, and atomically switches the catalog.
Writers coordinate through a maintenance lock. Do not edit SQLite directly.
If a backup is restored under a different memory root, run `rebuild-index`
before `doctor`: catalog source paths are absolute and are regenerated from the
restored immutable Markdown Events.

To return to the pre-v2 skill:

```bash
python3 "$PLM" rollback
```

Rollback first exports post-cutover v2 events in v1-compatible Markdown. Add
`--materialize-v1` only when those files must immediately enter the legacy
project directories.

## Release process

1. Run `python3 -m unittest discover -s tests -v` in the source repository.
2. Run `python3 scripts/plm.py install`; installation runs the bundled tests
   and then enters shadow mode.
3. Run installed smoke queries and `verify`. Verification executes the bundled
   tests and binds the report to the installed code fingerprint and Event count.
4. Run `cutover` only when every acceptance gate is true.
5. Confirm `config` reports `engine=v2`, `write_mode=v2`, then run `doctor`.

High-risk facts require verifiable evidence. Candidates that stop at
`needs_confirmation` can be reviewed explicitly:

```bash
python3 "$PLM" candidate-review --cwd "$PWD" --candidate-id '<id>' \
  --decision approve --evidence-kind user_confirmation \
  --evidence-json '{"confirmation":"User confirmed the current state"}'
```
