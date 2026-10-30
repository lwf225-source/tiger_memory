"""Synthetic local startup receipt. Never supplier evidence or production approval."""
import hashlib
import json
from pathlib import Path
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tests.test_embedding_budget import policy


def receipt_fixture(root, endpoint):
    root = Path(root)
    budget = PersistentEmbeddingBudget.initialize(root / "synthetic-budget", policy(
        endpoint=endpoint,
        region="dmxapi" if endpoint == "https://www.dmxapi.cn/v1/embeddings" else "cn-beijing",
        dimensions=1024, test_only=False,
        evidence_sha256=hashlib.sha256(b"SYNTHETIC-ONLY-NOT-APPROVAL").hexdigest()))
    receipt = root / "synthetic-receipt.json"
    receipt.write_text(json.dumps(budget.receipt()))
    receipt.chmod(0o600)
    return receipt, budget
