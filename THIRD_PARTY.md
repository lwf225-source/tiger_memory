# External components

The core Python runtime has no third-party runtime dependencies. Optional
components are installed separately; the tiger_memory wheel does not bundle their source,
binaries, model weights or user indexes.

## CodeGraph

- Upstream: https://github.com/colbymchenry/codegraph
- Package: `@colbymchenry/codegraph`
- Integration tested against: `1.5.0`
- Upstream license: MIT
- Copyright notice: Copyright (c) 2026 Colby Mchenry
- Versioned license: https://github.com/colbymchenry/codegraph/blob/v1.5.0/LICENSE
- Used through: an optional local CLI subprocess; existing hosts may instead
  call their own CodeGraph MCP integration.

CodeGraph provides code analysis and graph queries. tiger_memory's adapter provides the
memory/code response assembly and repository-snapshot checks. These are separate
components. The upstream license applies to CodeGraph, not automatically to tiger_memory.

The local acceptance environment used Python 3.9.6, CodeGraph 1.5.0 and the npm
distribution on macOS. Other versions/platforms require their own verification.
See `docs/CODEGRAPH.md` for the supported invocation and validation commands.

## Optional models and evaluation data

The optional embedding environment is pinned in `repro/model-requirements.lock`
(measured with Python 3.12.14 on macOS arm64). The BGE model identifier, fixed
revision and nine file hashes are recorded in `repro/bge-small-en-v1.5.json`.
These dependencies and models retain their own licenses. Model weights and
external evaluation datasets are not included in the Python distribution.
The code-memory comparison uses project-authored synthetic fixtures under
`examples/code_memory/`; `evaluation/project-docs-v1/source-snapshots/` contains
historical project-document fixtures, with their historical names preserved.

- Embedding model: https://huggingface.co/BAAI/bge-small-en-v1.5
- Sentence Transformers: https://github.com/UKPLab/sentence-transformers
- Transformers: https://github.com/huggingface/transformers
- PyTorch: https://github.com/pytorch/pytorch

The AML adapter retains comments crediting ideas associated with FlowGrid,
ReFind and Chronicle. These comments describe methodological influences;
this upload does not claim those algorithmic ideas as new contributions.
See `docs/AML_SUBMISSION.md` for the original project baseline and this upload's
changes. Optional reader/remote-LLM code paths are not enabled in the disclosed
default Add/Search configuration.

## Project license

The repository includes the Apache License, Version 2.0, in the root `LICENSE`.
This inventory does not change the separate licenses of external components.
