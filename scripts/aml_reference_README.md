# Local synthetic reference

`python -B -m tiger_memory.aml_reference --self-check` verifies the packaged
runtime/support hashes without reading credentials or contacting a model.
`--rehearse` uses a temporary store and a loopback synthetic embedding service.
It exercises lifecycle, durable cache reuse, recovery and diagnostic behavior.
The reference does not deploy a service or authorize paid calls. Its minimal
candidate metadata is non-authorizing and does not establish quality acceptance.
For the separate runtime snapshot, see `repro/runtime-20261030/README.md`.
