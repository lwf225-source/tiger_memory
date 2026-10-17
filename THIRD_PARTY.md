# Shipped-source and external-component inventory

The43 tiger_memory Python modules are copied byte-for-byte from the project's frozen88v3 runtime. Root LICENSE is the repository's Apache License2.0; existing source comments and attribution are retained. New round92 packaging scripts and documents are project distribution work under that same license. No original third-party license/NOTICE file was removed from any shipped source. NOTICE describes the changed packaging scope; runtime files are unchanged.

The wheel/sdist bundle no dependency wheels, copied third-party Python packages, binaries, model weights or datasets. Runtime candidate dependencies are Python standard library (including SQLite/FTS5) plus a separately installed Git executable. Those platform implementations retain their own licenses and are not redistributed here. The new PEP517 backend has no build-system requirements.

Optional runtime paths retained but not enabled by the Add/Search candidate:

- CodeGraph CLI integration: upstream https://github.com/colbymchenry/codegraph ; previously recorded integration1.5.0; upstream MIT, Copyright(c)2026 Colby Mchenry. No CodeGraph source/npm package is shipped. Its versioned license is https://github.com/colbymchenry/codegraph/blob/v1.5.0/LICENSE .
- MLX / MLX-LM optional local reader imports: https://github.com/ml-explore/mlx and https://github.com/ml-explore/mlx-lm . Neither package nor weights are shipped; their external licenses/terms apply if an operator separately enables them. No license grant for those packages is inferred from this repository's LICENSE.
- BGE/Sentence Transformers/Transformers/PyTorch references are historical optional baseline paths; no BGE model, lockfile environment or binaries are included in candidate92. Old BGE documents/benchmarks remain outside this distribution.
- Alibaba Cloud Model Studio text-embedding-v4 is an external API, not distributed code or weights. Service/model terms and authorized charges remain separate from Apache-2.0. No provider SDK is required or shipped. API source: https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api .

LongMemEval/LoCoMo and other evaluation data are not distributed; retained adapters are project code, not permission to redistribute third-party datasets. The only generated content used by packaged rehearsal is visibly named synthetic canary data authored in the project code.

The project previously credited FlowGrid, ReFind and Chronicle as methodological influences. Attribution is retained in docs/METHOD.md and NOTICE; no novelty or license claim over their algorithms or implementations is made.
