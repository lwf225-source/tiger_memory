from __future__ import annotations

import hashlib
import json
import unittest

from tiger_memory.passages import expand_passage, locate_passage, neighbor_passages, split_passages, validate_passage


class PassageTests(unittest.TestCase):
    def assert_exact_coverage(self, body, passages, maximum):
        covered = [False] * len(body)
        for passage in passages:
            self.assertEqual(passage["text"], body[passage["start"]:passage["end"]])
            self.assertLessEqual(len(passage["text"]), maximum)
            self.assertEqual(passage["body_sha256"], hashlib.sha256(body.encode("utf-8")).hexdigest())
            validate_passage(body, passage, "event-fixture")
            for position in range(passage["start"], passage["end"]):
                covered[position] = True
        self.assertTrue(all(covered))

    def test_document_boundaries_preserve_raw_text_and_all_characters(self):
        body = "# 说明\n\n" + "第一段包含部署条件。" * 12 + "\n\n" + "尚未通过真机验收，不得发布。" * 10 + "\n末尾。\n"
        passages = split_passages(body, "event-fixture", max_chars=90, overlap_chars=15)
        self.assert_exact_coverage(body, passages, 90)
        self.assertTrue(all(item["speaker"] == "unknown" and item["source_date"] == "unknown" for item in passages))
        self.assertTrue(any("不得发布" in item["text"] for item in passages))
        self.assertEqual(passages, split_passages(body, "event-fixture", 90, 15))

    def test_single_giant_line_and_unicode_are_bounded_without_encoding_damage(self):
        body = "中文🙂é𠮷条件不得发布" * 150
        passages = split_passages(body, "event-fixture", max_chars=53, overlap_chars=9)
        self.assert_exact_coverage(body, passages, 53)
        self.assertTrue(all(item["text"].encode("utf-8").decode("utf-8") == item["text"] for item in passages))
        self.assertTrue(all(item["line_start"] == 1 and item["line_end"] == 1 for item in passages))

    def test_newline_line_ranges_are_one_based_and_preserve_crlf(self):
        body = "one\r\ntwo\r\nthree\n"
        passages = split_passages(body, "event-fixture", max_chars=6, overlap_chars=0)
        self.assert_exact_coverage(body, passages, 6)
        self.assertEqual(passages[0]["text"], "one\r\n")
        self.assertEqual((passages[0]["line_start"], passages[0]["line_end"]), (1, 1))
        self.assertEqual(passages[1]["line_start"], 2)

    def test_explicit_roles_and_dates_survive_long_turn_windows(self):
        body = "Session date: 2026/09/11 (Friday)\nUser: " + "这次只要求本地验证。" * 35 + "\nAssistant: 尚未部署。\n"
        passages = split_passages(body, "event-fixture", max_chars=75, overlap_chars=10)
        self.assert_exact_coverage(body, passages, 75)
        users = [item for item in passages if item["speaker"] == "user"]
        self.assertGreater(len(users), 1)
        self.assertEqual({item["unit_index"] for item in users}, {0})
        self.assertTrue(all(item["source_date"] == "2026/09/11 (Friday)" for item in passages))
        self.assertEqual(passages[-1]["speaker"], "assistant")

    def test_conversation_json_offsets_do_not_reencode_or_lose_speakers(self):
        turns = [{"role": "user", "content": "仅本地，不能发布。\n" + "长说明🙂" * 35},
                 {"role": "assistant", "content": "明白。"}]
        body = "Session date: 2023-05-30 (Tue) 14:00\n" + json.dumps(turns, ensure_ascii=False)
        passages = split_passages(body, "event-fixture", max_chars=80, overlap_chars=12)
        self.assert_exact_coverage(body, passages, 80)
        self.assertEqual(passages[0]["speaker"], "user")
        self.assertEqual(passages[-1]["speaker"], "assistant")
        self.assertEqual({item["source_date"] for item in passages}, {"2023-05-30 (Tue) 14:00"})
        self.assertTrue(any("\\n" in item["text"] for item in passages))

    def test_unknown_schema_does_not_invent_role_or_date(self):
        body = 'Alice yesterday\n[{"speaker": "user", "message": "不发布"}]'
        passages = split_passages(body, "event-fixture", 40, 5)
        self.assert_exact_coverage(body, passages, 40)
        self.assertEqual({item["speaker"] for item in passages}, {"unknown"})
        self.assertEqual({item["source_date"] for item in passages}, {"unknown"})

    def test_fenced_examples_are_not_treated_as_real_conversation_headers(self):
        body = "示例文本\n```text\nUser: 这只是示例\nDate: 2099-01-01\n```\n后续说明\n"
        passages = split_passages(body, "event-fixture", 40, 5)
        self.assert_exact_coverage(body, passages, 40)
        self.assertEqual({item["speaker"] for item in passages}, {"unknown"})
        self.assertEqual({item["source_date"] for item in passages}, {"unknown"})

    def test_markdown_role_headings_and_date_changes_are_explicit(self):
        body = "## user [2026-01-01]\n本地测试。\n## assistant [2026-01-02]\n不得发布。"
        passages = split_passages(body, "event-fixture")
        self.assertEqual([item["speaker"] for item in passages], ["user", "assistant"])
        self.assertEqual([item["source_date"] for item in passages], ["2026-01-01", "2026-01-02"])

    def test_chinese_role_headers_do_not_require_an_ascii_space(self):
        body = "日期：2026-09-11\n用户：仅本地。\n助手：尚未发布。"
        passages = split_passages(body, "event-fixture")
        self.assertEqual([item["speaker"] for item in passages], ["user", "assistant"])
        self.assertEqual({item["source_date"] for item in passages}, {"2026-09-11"})
        self.assert_exact_coverage(body, passages, 900)

    def test_parent_or_body_change_changes_stable_passage_ids(self):
        original = split_passages("abcdef", "event-fixture", 4, 1)
        self.assertNotEqual(original[0]["passage_id"], split_passages("abcdeg", "event-fixture", 4, 1)[0]["passage_id"])
        self.assertNotEqual(original[0]["passage_id"], split_passages("abcdef", "other-event", 4, 1)[0]["passage_id"])
        self.assertEqual(original, split_passages("abcdef", "event-fixture", 4, 1))

    def test_hash_offsets_text_metadata_and_identity_tampering_are_rejected(self):
        body = "User: only local.\nAssistant: do not deploy."
        passage = split_passages(body, "event-fixture")[0]
        for field, replacement in (("body_sha256", "0" * 64), ("start", -1), ("end", len(body) + 1),
                                   ("text", "invented"), ("speaker", "assistant"), ("source_date", "2099"),
                                   ("line_start", 99), ("line_end", True), ("unit_index", False),
                                   ("passage_id", "forged"), ("algorithm", "other")):
            bad = dict(passage, **{field: replacement})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_passage(body, bad)
        with self.assertRaises(ValueError):
            validate_passage(body + "!", passage)
        with self.assertRaises(ValueError):
            validate_passage(body, passage, "other-event")

    def test_locate_returns_exact_canonical_arbitrary_span_and_rejects_bad_offsets(self):
        body = "User: 条件是仅本地。\nAssistant: 不得发布。\n"
        original = split_passages(body, "event-fixture")[1]
        self.assertEqual(locate_passage(body, "event-fixture", original["start"], original["end"]), original)
        located = locate_passage(body, "event-fixture", 2, len(body) - 2)
        self.assertEqual(located["text"], body[2:-2])
        self.assertEqual(located["speaker"], "unknown")
        self.assertEqual(validate_passage(body, located, "event-fixture"), located)
        # Caller-owned annotations cannot contaminate the canonical result.
        self.assertEqual(validate_passage(body, dict(located, score=99)), located)
        for start, end in ((-1, 2), (0, len(body) + 1), (4, 4), (8, 2), (True, 2), (0, 1.0)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                locate_passage(body, "event-fixture", start, end)

    def test_expansion_is_bounded_exact_and_marks_mixed_speakers_unknown(self):
        body = "User: 条件是仅本地。\nAssistant: 尚未真机验证，不得发布。\n"
        passages = split_passages(body, "event-fixture")
        expanded = expand_passage(body, passages[1], before_chars=99, after_chars=99, max_chars=len(body))
        self.assertEqual(expanded["text"], body)
        self.assertEqual(expanded["speaker"], "unknown")
        self.assertEqual(expanded["unit_index"], -1)
        self.assertEqual(validate_passage(body, expanded), expanded)
        limited = expand_passage(body, passages[1], before_chars=100, max_chars=len(passages[1]["text"]) + 3)
        self.assertEqual(len(limited["text"]), len(passages[1]["text"]) + 3)
        with self.assertRaises(ValueError):
            expand_passage(body, passages[1], max_chars=1)

    def test_bounded_neighbors_include_conditions_and_preserve_each_source_span(self):
        body = "User: 只执行本地验收。\nAssistant: 已构建完成。\nUser: 不得发布，真机还没测。\n"
        passages = split_passages(body, "event-fixture")
        nearby = neighbor_passages(body, passages, passages[1]["passage_id"], radius=1, max_chars=len(body))
        self.assertEqual(nearby, passages)
        self.assertTrue(any("不得发布" in item["text"] for item in nearby))
        only_match = neighbor_passages(body, passages, passages[1]["passage_id"], radius=1, max_chars=len(passages[1]["text"]))
        self.assertEqual(only_match, [passages[1]])
        with self.assertRaises(ValueError):
            neighbor_passages(body, passages, "missing")
        with self.assertRaises(ValueError):
            neighbor_passages(body, passages + split_passages(body, "other-event"), passages[1]["passage_id"])

    def test_empty_whitespace_and_invalid_window_parameters(self):
        self.assertEqual(split_passages("", "event-fixture"), [])
        whitespace = split_passages(" \n\t", "event-fixture", 2, 0)
        self.assert_exact_coverage(" \n\t", whitespace, 2)
        for maximum, overlap in ((0, 0), (10, 10), (10, -1), (True, 0), (10, False)):
            with self.subTest(maximum=maximum, overlap=overlap), self.assertRaises(ValueError):
                split_passages("abc", "event-fixture", maximum, overlap)
        with self.assertRaises(ValueError):
            split_passages("abc", "")
        with self.assertRaises(ValueError):
            split_passages("\ud800", "event-fixture")
