from __future__ import annotations

import concurrent.futures
import os
import threading
import unittest
from pathlib import Path

from tiger_memory.database import connect
from tiger_memory.service import write_event
from tests.common import MemoryTestCase


def _process_write(args):
    base, project, index = args
    os.environ["TIGER_MEMORY_DIR"] = base
    from tiger_memory.service import write_event as worker_write
    event = worker_write(
        Path(project), "并发记忆 %03d" % index, "并发写入内容 %03d" % index,
        tags="concurrency", idempotency_key="worker-%03d" % index,
    )
    return event.event_id


class ConcurrencyTests(MemoryTestCase):
    def test_concurrent_connect_on_fresh_database(self):
        # No pre-initialization: racing connect() calls must all survive the
        # one-time source_event_id migration on a fresh catalog.
        errors = []

        def worker():
            try:
                conn = connect()
                conn.close()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(errors, [])
        conn = connect()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        conn.close()
        self.assertIn("source_event_id", columns)

    def test_one_hundred_process_writes_and_retries(self):
        write_event(self.project, "初始化", "先初始化数据库。", idempotency_key="init")
        tasks = [(str(self.base), str(self.project), index) for index in range(100)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            ids = list(executor.map(_process_write, tasks))
        self.assertEqual(len(set(ids)), 100)
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            retry_ids = list(executor.map(_process_write, tasks))
        self.assertEqual(ids, retry_ids)
        conn = connect()
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        conn.close()
        self.assertEqual(count, 101)
        self.assertEqual(fts, 101)


if __name__ == "__main__":
    unittest.main()

