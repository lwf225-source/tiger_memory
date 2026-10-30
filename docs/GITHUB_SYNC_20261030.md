# Source synchronization — 2026-10-30

This publication contains the current 2.2.0a5 development source, explicit Active
Memory, standalone synthetic tests, examples, skill resources and packaging tools.
A separate portable runtime source snapshot is in `repro/runtime-20261030`.
Its entry uses environment configuration and its manifest binds public source
bytes. The root AML entry preserves its earlier budget-bound behavior.

Private memory records, credentials, experimental datasets, operational output,
deployment history, live configuration identities and internal provenance paths
are excluded. Historical public documentation is retained from the earlier public
source revision; it is not a statement of current deployment status.

## Verification

- Active Memory acceptance: 22/22 passed.
- Separate runtime transport checks: 5/5 passed with synthetic/loopback transport.
- Offline sdist install: 9/9 passed.
- Complete source run: 758 tests, with one installed-Skill path issue found.
  After the fix, all five affected checks and the complete installed bundle suite
  passed. The outer full run was not repeated after this focused correction.
- The trace resolver also ran from an actual wheel under isolated Python.
- Credential scan matches were reviewed as synthetic negative-test strings.

Historical experiment tests requiring private artifacts are excluded and are
not counted as passes. Source publication does not deploy a service or submit
an official evaluation.
