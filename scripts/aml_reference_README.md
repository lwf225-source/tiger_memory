# Candidate92 packaged synthetic reference

The unchanged88v3 aml_reference runtime verifies the new selected manifest/support closure.
Run `python -I -B -m tiger_memory.aml_reference --self-check` without real configuration.
Run `python -I -B -m tiger_memory.aml_reference --rehearse` only for disposable loopback fake data.
Neither mode reads existing credentials/receipt or grants production access. Rehearsal uses
an explicit test-only fake budget and fake endpoint; production rejects that receipt.
Service-object restarts are not process kills. See packaged README and recovery guide.
Old75 zero-lost-source gate remains FAIL; candidate UNADOPTED/DECISION_PENDING.
The new package is separately verified; older715/L3 evidence only covers its unchanged scope.
