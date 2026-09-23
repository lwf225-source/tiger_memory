from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tiger_memory.evaluation import SOURCE_FIELDS, load_dataset
from tiger_memory.model import SCOPES

FOLDER = Path(__file__).resolve().parents[1] / 'evaluation/long-context-v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class LongContextDatasetTests(unittest.TestCase):
    def test_counts_lengths_and_scope_schema(self):
        combined = []
        for split in ('dev','heldout'):
            data = load_dataset(FOLDER, split)
            self.assertEqual(len(data['questions']), 12)
            self.assertEqual(len(data['sources']), 6)
            self.assertEqual(sum(row['unanswerable'] for row in data['questions']), 2)
            self.assertEqual(sum(row['scope']=='session' for row in data['questions']), 2)
            combined += data['sources']
            self.assertTrue(data['manifest']['synthetic'])
            self.assertFalse(data['manifest']['real_user_query_sample'])
            self.assertFalse(data['manifest']['selected_using_predictions'])
        for source in combined:
            self.assertFalse(set(source)-SOURCE_FIELDS)
            self.assertIn(source['scope'], SCOPES)
            self.assertTrue(4000 <= len(source['body']) <= 12000, source['source_id'])
        self.assertEqual(len({digest(row['body']) for row in combined}),12)

    def test_spans_are_exact_hash_bound_in_scope_and_cover_all_answerable_rows(self):
        for split in ('dev','heldout'):
            data = load_dataset(FOLDER, split)
            sidecar = read(FOLDER / ('evidence-spans.'+split+'.json'))
            self.assertTrue(sidecar['retrieval_use_forbidden'])
            self.assertEqual(sidecar['sources_sha256'], data['manifest']['sha256']['sources.json'])
            self.assertEqual(sidecar['questions_sha256'], data['manifest']['sha256']['questions.'+split+'.json'])
            by_id = {row['source_id']:row for row in data['sources']}
            mapping = {row['question_id']:row for row in sidecar['questions']}
            self.assertEqual(set(mapping),{row['question_id'] for row in data['questions']})
            for question in data['questions']:
                entry = mapping[question['question_id']]
                self.assertEqual(entry['unanswerable'],question['unanswerable'])
                self.assertEqual({span['source_id'] for span in entry['spans']},set(question['required_source_ids']))
                if question['unanswerable']:
                    self.assertFalse(entry['spans'])
                    self.assertTrue(entry['absence_check_source_ids'])
                else:
                    self.assertGreaterEqual(len(entry['spans']),2)
                    self.assertFalse(entry['absence_check_source_ids'])
                for span in entry['spans']:
                    source = by_id[span['source_id']]
                    self.assertEqual((source['scope'],source['scope_id']),(question['scope'],question['scope_id']))
                    self.assertTrue(type(span['start']) is int and type(span['end']) is int)
                    self.assertTrue(0 <= span['start'] < span['end'] <= len(source['body']))
                    quote = source['body'][span['start']:span['end']]
                    self.assertTrue(digest(quote)==span['quote_sha256'],question['question_id'])

    def test_mid_tail_pronoun_multisource_and_scope_distractor_coverage(self):
        for split in ('dev','heldout'):
            data = load_dataset(FOLDER,split)
            by_id = {row['source_id']:row for row in data['sources']}
            mapping = {row['question_id']:row for row in read(FOLDER/('evidence-spans.'+split+'.json'))['questions']}
            ratios = [span['start']/len(by_id[span['source_id']]['body']) for row in mapping.values() for span in row['spans']]
            self.assertTrue(any(.3 <= ratio < .7 for ratio in ratios))
            self.assertTrue(any(ratio >= .85 for ratio in ratios))
            self.assertGreaterEqual(sum(len(row['required_source_ids'])>1 for row in data['questions']),1)
            pronouns = [row for row in data['questions'] if row['category']=='pronoun_dependency']
            self.assertEqual(len(pronouns),2)
            for row in pronouns:
                spans = mapping[row['question_id']]['spans']
                self.assertEqual(spans[0]['source_id'],spans[1]['source_id'])
                self.assertEqual(spans[0]['end']+2,spans[1]['start'])
            for project in {row['project'] for row in data['sources']}:
                rows = [row for row in data['sources'] if row['project']==project]
                self.assertEqual({row['scope'] for row in rows},{'project','session'})

    def test_parent_groups_and_question_labels_do_not_leak_into_sources(self):
        manifest = read(FOLDER/'manifest.json')
        groups = manifest['parent_event_groups']
        self.assertEqual(len({row['parent_event_group'] for row in groups}),4)
        self.assertEqual(len({row['project'] for row in groups}),4)
        all_sources = read(FOLDER/'sources.json')
        project_splits = {row['project']:row['split'] for row in groups}
        for source in all_sources:
            self.assertEqual(source['split'],project_splits[source['project']])
        source_text = '\n'.join(row['body'] for row in all_sources)
        for split in ('dev','heldout'):
            for question in load_dataset(FOLDER,split)['questions']:
                self.assertNotIn(question['question_id'],source_text)
                self.assertTrue(question['question'] not in source_text,question['question_id'])
                self.assertNotIn('evidence_keys',question)
                self.assertGreaterEqual(len(question['acceptable_answers']),2)
                self.assertTrue(question['required_points'])
                for point in question['required_points']:
                    self.assertTrue(any(form.casefold() in question['answer'].casefold() for form in point['any_of']),question['question_id'])

    def test_deterministic_rebuild_matches_every_frozen_artifact(self):
        specification = importlib.util.spec_from_file_location('long_context_builder',FOLDER/'build_dataset.py')
        builder = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(builder)
        with tempfile.TemporaryDirectory(prefix='plm-long-context-fixture-') as temporary:
            target = Path(temporary)
            builder.build(target)
            for name in ('manifest.json','sources.json','questions.dev.json','questions.heldout.json',
                         'evidence-spans.dev.json','evidence-spans.heldout.json','provenance.json'):
                self.assertTrue(hashlib.sha256((target/name).read_bytes()).digest()==hashlib.sha256((FOLDER/name).read_bytes()).digest(),name)

    def test_provenance_hashes_and_cross_split_text_separation(self):
        provenance = read(FOLDER/'provenance.json')
        self.assertEqual(provenance['builder_sha256'],hashlib.sha256((FOLDER/'build_dataset.py').read_bytes()).hexdigest())
        self.assertEqual(provenance['dataset_manifest_sha256'],digest((FOLDER/'manifest.json').read_text()))
        for name, expected in provenance['sidecar_sha256'].items():
            self.assertEqual(expected,digest((FOLDER/name).read_text()))
        def grams(text):
            lowered = ''.join(text.casefold().split())
            return {lowered[index:index+12] for index in range(len(lowered)-11)}
        dev = [grams(row['body']) for row in load_dataset(FOLDER,'dev')['sources']]
        heldout = [grams(row['body']) for row in load_dataset(FOLDER,'heldout')['sources']]
        for left in dev:
            for right in heldout:
                self.assertLess(len(left & right)/max(1,len(left|right)),.2)
