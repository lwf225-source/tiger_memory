"""Narrative-time tagging tests (batch 11): deterministic write-side date
extraction and its opt-in wiring into the AML adapter Add path.

Discipline under test:
- the extractor takes only explicit calendar expressions; invalid dates and
  anchor-less bare month names are rejected, and extraction never raises;
- relative phrases contribute their absolute anchor ("two weeks before
  August 11" -> "August 11") but are never resolved against record time;
- the injection point is event *tags* (metadata): the immutable body is
  byte-identical with the switch on or off, so Search evidence stays verbatim;
- switch OFF is byte-identical to the pre-batch-11 write path (tags == ["aml"]).
"""

from __future__ import annotations

import json
from pathlib import Path

from tiger_memory import aml_adapter
from tiger_memory.events import parse_event
from tiger_memory.narrative_time import MAX_MENTIONS, MAX_TAGS, extract_mentioned_dates, mentioned_date_tags
from tests.common import MemoryTestCase


class ExtractorTests(MemoryTestCase):
    """Pure extractor behavior (no storage involved)."""

    def test_iso_date(self):
        items = extract_mentioned_dates("we moved on 2023-05-08 and never looked back")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "date")
        self.assertEqual(items[0]["iso"], "2023-05-08")
        tags = mentioned_date_tags("we moved on 2023-05-08 and never looked back")
        self.assertIn("2023-05-08", tags)
        self.assertIn("May 8, 2023", tags)
        self.assertIn("May 2023", tags)
        self.assertIn("spring 2023", tags)
        self.assertIn("2023", tags)

    def test_month_day_year(self):
        tags = mentioned_date_tags("my trip on August 11, 2023 was great")
        self.assertIn("2023-08-11", tags)
        self.assertIn("August 11, 2023", tags)
        self.assertIn("August 2023", tags)
        self.assertIn("summer 2023", tags)

    def test_day_first_and_ordinal(self):
        for text in ("11 August 2023", "the 11th of August, 2023"):
            tags = mentioned_date_tags(text)
            self.assertIn("2023-08-11", tags, msg=text)
            # The richer rule claims the whole span: no coarser duplicate.
            self.assertNotIn("2023-08", tags, msg=text)

    def test_month_year(self):
        items = extract_mentioned_dates("the March 2026 report")
        self.assertEqual([item["kind"] for item in items], ["month_year"])
        tags = mentioned_date_tags("the March 2026 report")
        self.assertIn("March 2026", tags)
        self.assertIn("2026-03", tags)
        self.assertIn("2026", tags)

    def test_month_day_without_year(self):
        items = extract_mentioned_dates("two weeks before August 11 we met")
        self.assertEqual([item["kind"] for item in items], ["month_day"])
        self.assertEqual(mentioned_date_tags("two weeks before August 11 we met"), ["August 11"])

    def test_season_year(self):
        tags = mentioned_date_tags("in summer 2021 we drove to Utah")
        self.assertIn("summer 2021", tags)
        self.assertIn("2021", tags)
        tags = mentioned_date_tags("the autumn of 2021")
        self.assertIn("fall 2021", tags)
        self.assertIn("autumn 2021", tags)

    def test_season_bridging_from_full_date(self):
        # A July date becomes findable by a "summer" query.
        tags = mentioned_date_tags("we left on July 3, 2021")
        self.assertIn("summer 2021", tags)

    def test_bare_year(self):
        self.assertEqual(mentioned_date_tags("I visited Kenya in 2021"), ["2021"])

    def test_chinese_forms(self):
        tags = mentioned_date_tags("2023年8月11日见面")
        self.assertIn("2023-08-11", tags)
        tags = mentioned_date_tags("截至2026年3月")
        self.assertIn("March 2026", tags)

    def test_rejections(self):
        for text in ("error code 2026-13-45 broke",   # invalid month/day; year stays masked
                     "Feb 30, 2023",                  # invalid day; year stays masked
                     "the March report was fine",     # bare month, no year/day anchor
                     "in 1876 they arrived",          # out of range
                     ""):
            self.assertEqual(mentioned_date_tags(text), [], msg=text)

    def test_dedupe_and_order(self):
        tags = mentioned_date_tags("in 2021 and again in 2021; later on 2021-05-08")
        self.assertEqual(tags, ["2021", "2021-05-08", "May 8, 2021", "May 2021", "spring 2021"])

    def test_caps(self):
        text = " ".join("event on %04d-01-%02d." % (2000 + i, i + 1) for i in range(12))
        items = extract_mentioned_dates(text)
        self.assertLessEqual(len(items), MAX_MENTIONS)
        self.assertLessEqual(len(mentioned_date_tags(text)), MAX_TAGS)

    def test_never_raises_on_odd_input(self):
        for value in (None, 123, ["2021"], "在2099年99月", "\x00\xff 2023-99-99"):
            self.assertEqual(extract_mentioned_dates(value), [])
            self.assertEqual(mentioned_date_tags(value), [])


class AdapterNarrativeTimeTests(MemoryTestCase):
    """Write-side wiring: tags carry the mentions, the body never changes."""

    EVIDENCE = "I went on a safari in Kenya back in 2021, it was unforgettable."

    def setUp(self):
        super().setUp()
        self._session_rrf = aml_adapter.AML_SESSION_RRF
        # These tests isolate narrative-time tags from the separately tested
        # 1.3.1 session-RRF routing tag.
        aml_adapter.AML_SESSION_RRF = False

    def tearDown(self):
        aml_adapter.AML_NARRATIVE_TIME = False
        aml_adapter.AML_SESSION_RRF = self._session_rrf
        super().tearDown()

    def _add(self, user_id, texts):
        return aml_adapter.add({
            "request_id": "r1", "user_id": user_id, "session_id": "s1",
            "messages": [
                {"role": "user", "timestamp": 1704067200000 + index * 60000, "content": text}
                for index, text in enumerate(texts)
            ],
        })

    def _events(self, user_id):
        from tiger_memory.model import v2_root
        return [event for event in
                (parse_event(path) for path in sorted(v2_root().glob("projects/*/events/*/*/*.md")))
                if event.metadata["scope_id"] == user_id]

    def test_default_off_tags_exactly_source(self):
        self._add("u1", [self.EVIDENCE])
        (event,) = self._events("u1")
        self.assertEqual(event.metadata["tags"], ["aml"])
        self.assertEqual(event.body, self.EVIDENCE)
        # OFF is byte-identical to the pre-batch-11 file layout: no date tag
        # text anywhere in the serialized frontmatter.
        raw = event.path.read_text(encoding="utf-8")
        self.assertIn('tags: ["aml"]', raw)
        self.assertNotIn("2021-01", raw.split("---")[1])

    def test_switch_on_tags_dates_body_untouched(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("u2", [self.EVIDENCE])
        (event,) = self._events("u2")
        self.assertEqual(event.body, self.EVIDENCE)  # immutable evidence body
        tags = event.metadata["tags"]
        self.assertEqual(tags[0], "aml")
        self.assertIn("2021", tags)

    def test_switch_on_full_date_forms(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("u3", ["we hiked on July 3, 2021 together"])
        (event,) = self._events("u3")
        tags = event.metadata["tags"]
        for expected in ("2021-07-03", "July 3, 2021", "July 2021", "summer 2021", "2021"):
            self.assertIn(expected, tags)

    def test_no_date_message_gets_no_extra_tags(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("u4", ["we talked about hiking gear"])
        (event,) = self._events("u4")
        self.assertEqual(event.metadata["tags"], ["aml"])

    def test_search_evidence_body_verbatim(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("u5", [self.EVIDENCE])
        response = aml_adapter.search({"query": "Which country did James visit in 2021?",
                                       "user_id": "u5", "top_k": 5})
        bodies = [item["content"] for item in response["data"]]
        self.assertIn(self.EVIDENCE, bodies)  # exact original text, no annotation

    def test_retrieval_gain_date_anchored_query(self):
        distractors = [
            "James told me about the country he wants to visit next year.",
            "Which country should we visit for the next trip? James had many ideas.",
            "James said the country he would visit must have good food.",
        ]
        self._add("uoff", distractors + [self.EVIDENCE])
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("uon", distractors + [self.EVIDENCE])
        aml_adapter.AML_NARRATIVE_TIME = False
        query = "Which country did James visit in 2021?"

        def rank(user_id):
            response = aml_adapter.search({"query": query, "user_id": user_id, "top_k": 10})
            bodies = [item["content"] for item in response["data"]]
            return bodies.index(self.EVIDENCE) if self.EVIDENCE in bodies else None

        off_rank, on_rank = rank("uoff"), rank("uon")
        self.assertIsNotNone(off_rank)
        self.assertIsNotNone(on_rank)
        # The narrative-time tag boost lifts the date-mentioning evidence.
        self.assertLess(on_rank, off_rank)

    def test_idempotent_replay_unchanged(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        self._add("u6", [self.EVIDENCE])
        first = self._events("u6")[0].path.read_bytes()
        self._add("u6", [self.EVIDENCE])  # same request_id replayed
        events = self._events("u6")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].path.read_bytes(), first)

    def test_add_result_contract_unchanged(self):
        aml_adapter.AML_NARRATIVE_TIME = True
        result = self._add("u7", [self.EVIDENCE])
        self.assertEqual(result, {"success": True, "request_id": "r1", "user_id": "u7", "session_id": "s1"})
