from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tiger_memory.evaluation import load_dataset, write_dataset
from tiger_memory.model import PLMError
from tiger_memory.span_evaluation import load_span_gold, score_span_coverage


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SpanEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='plm-span-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.body = '记录开始🙂。\n甲：现在说的是顶灯，不是台面。\n乙：它必须关闭；台面仍可使用。\n后记：这一限制仅适用于周五。'
        self.source = {'source_id':'parent-a','project':'fixture-project','split':'dev','title':'Record','body':self.body,'scope':'project','scope_id':''}
        questions = [
            {'question_id':'q1','project':'fixture-project','split':'dev','question':'What must be off, and under which condition?','answer':'Fixture answer','category':'pronoun','unanswerable':False,'required_source_ids':['parent-a']},
            {'question_id':'q2','project':'fixture-project','split':'dev','question':'What is the unknown price?','answer':'Unknown','category':'absence','unanswerable':True,'required_source_ids':[]}]
        write_dataset(self.root,[self.source],questions,{'dataset_id':'span-fixture','origin':'synthetic','splits':['dev']})
        self.dataset = load_dataset(self.root,'dev')
        quotes = ['甲：现在说的是顶灯，不是台面。','乙：它必须关闭；台面仍可使用。','后记：这一限制仅适用于周五。']
        self.required = [self.span(quote, 'point-'+str(index)) for index,quote in enumerate(quotes)]
        self.sidecar = {'schema_version':1,'dataset_id':'span-fixture','split':'dev','retrieval_use_forbidden':True,
            'offset_unit':'Unicode code points; start inclusive/end exclusive; hash of quote.encode(utf-8)',
            'sources_sha256':self.dataset['manifest']['sha256']['sources.json'],
            'questions_sha256':self.dataset['manifest']['sha256']['questions.dev.json'],
            'questions':[{'question_id':'q1','unanswerable':False,'spans':self.required,'absence_check_source_ids':[]},
                         {'question_id':'q2','unanswerable':True,'spans':[],'absence_check_source_ids':['parent-a']}]}
        self.path = self.root/'evidence-spans.dev.json'
        self.save()
        self.gold,self.fingerprint = load_span_gold(self.dataset,'dev',self.root)

    def save(self):
        self.path.write_text(json.dumps(self.sidecar,ensure_ascii=False),encoding='utf-8')

    def span(self,quote,identity='segment'):
        start = self.body.index(quote)
        return {'evidence_id':identity,'source_id':'parent-a','start':start,'end':start+len(quote),'quote_sha256':sha(quote)}

    def case(self,body=None,spans=None,passage=False):
        body = self.body if body is None else body
        spans = [self.span(self.body)] if spans is None else spans
        sid = 'passage-a' if passage else 'parent-a'
        metadata = {'source_id':sid,'body_sha256':sha(self.body),'presented_body_sha256':sha(body),'spans':copy.deepcopy(spans)}
        if passage:
            metadata['parent_source_id']='parent-a'
        return {'source_ids':[sid],'sources':[metadata], 'messages':[{'role':'system','content':'Fixture only'},
            {'role':'user','content':json.dumps({'question':'Fixture','sources':[{'source_id':sid,'body':body}]},ensure_ascii=False)}]}

    def score(self,case):
        return score_span_coverage('q1',case,self.gold,{'parent-a':self.source})

    def test_loader_binds_hashes_unicode_offsets_and_missing_is_unmeasured(self):
        self.assertEqual(len(self.gold),2)
        self.assertEqual(self.fingerprint,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.dataset['root']=self.root
        self.assertEqual(load_span_gold(self.dataset,'dev')[1],self.fingerprint)
        self.assertEqual(load_span_gold({},'dev'),({},None))
        self.assertEqual(load_span_gold(self.dataset,'heldout',self.root),({},None))
        self.assertNotEqual(self.required[0]['start'],len(self.body[:self.required[0]['start']].encode('utf-8')))
        for bad in ('sources_sha256','questions_sha256'):
            original=self.sidecar[bad]
            self.sidecar[bad]='0'*64
            self.save()
            with self.assertRaises(PLMError): load_span_gold(self.dataset,'dev',self.root)
            self.sidecar[bad]=original
        self.sidecar['questions'][0]['spans'][0]['start'] += 1
        self.save()
        with self.assertRaises(PLMError): load_span_gold(self.dataset,'dev',self.root)

    def test_full_and_passage_payloads_cover_all_required_spans(self):
        for passage in (False,True):
            result = self.score(self.case(passage=passage))
            self.assertEqual(result['covered'],3)
            self.assertEqual(result['required'],3)
            self.assertTrue(result['full_coverage'])
        start=self.required[0]['start']
        quote=self.body[start:]
        self.assertTrue(self.score(self.case(quote,[self.span(quote)],True))['full_coverage'])

    def test_parent_id_hit_without_antecedent_and_tail_is_not_full(self):
        span=self.required[1]
        quote=self.body[span['start']:span['end']]
        result=self.score(self.case(quote,[span]))
        self.assertEqual(result['covered'],1)
        self.assertFalse(result['full_coverage'])
        self.assertEqual([row['covered'] for row in result['per_span']],[False,True,False])

    def test_forged_span_hash_body_and_unseen_text_never_earn_coverage(self):
        fake=self.case('Only an unrelated heading')
        self.assertEqual(self.score(fake)['covered'],0)
        self.assertEqual(self.score(fake)['invalid_advertised_spans'],1)
        fake=self.case()
        fake['sources'][0]['body_sha256']='0'*64
        self.assertEqual(self.score(fake)['covered'],0)
        fake=self.case()
        fake['sources'][0]['spans'][0]['quote_sha256']='0'*64
        self.assertEqual(self.score(fake)['covered'],0)
        fake=self.case(self.body.replace('关闭','开启'),[self.span(self.body)])
        self.assertEqual(self.score(fake)['covered'],0)
        fake=self.case()
        fake['messages']=[]
        self.assertEqual(self.score(fake)['covered'],0)

    def test_scope_and_changed_parent_are_rejected_even_if_ids_match(self):
        changed=dict(self.source,scope='session',scope_id='workshop')
        result=score_span_coverage('q1',self.case(),self.gold,{'parent-a':changed})
        self.assertEqual(result['covered'],0)
        changed=dict(self.source,body=self.body+' Extra material')
        case=self.case()
        case['sources'][0]['body_sha256']=sha(changed['body'])
        self.assertEqual(score_span_coverage('q1',case,self.gold,{'parent-a':changed})['covered'],0)

    def test_missing_metadata_and_duplicate_payload_ids_are_conservative(self):
        case=self.case()
        case['sources'][0]['spans']=[]
        self.assertEqual(self.score(case)['covered'],0)
        case=self.case()
        payload=json.loads(case['messages'][1]['content'])
        payload['sources'].append(payload['sources'][0])
        case['messages'][1]['content']=json.dumps(payload)
        self.assertEqual(self.score(case)['covered'],0)
        case=self.case()
        case['sources'].append(copy.deepcopy(case['sources'][0]))
        self.assertEqual(self.score(case)['covered'],0)

    def test_unanswerable_is_not_auto_full_coverage(self):
        for qid in ('q2','missing'):
            result=score_span_coverage(qid,self.case(),self.gold,{'parent-a':self.source})
            self.assertEqual(result['status'],'not_measured')
            self.assertIsNone(result['full_coverage'])
            self.assertIsNone(result['coverage'])

    def test_repeated_short_quote_is_not_bound_to_a_claimed_location(self):
        body='prefix Yes. middle Yes. suffix'
        source=dict(self.source,body=body)
        start=body.index('Yes.')
        span={'start':start,'end':start+4,'quote_sha256':sha('Yes.')}
        gold={'q':{'unanswerable':False,'spans':[dict(span,evidence_id='duplicate',source_id='parent-a')],
                  '_project':'fixture-project','_scope':'project','_scope_id':'','_source_hashes':{'parent-a':sha(body)}}}
        case=self.case('Yes.',[span])
        case['sources'][0]['body_sha256']=sha(body)
        result=score_span_coverage('q',case,gold,{'parent-a':source})
        self.assertEqual(result['covered'],0)
        self.assertEqual(result['invalid_advertised_spans'],1)

    def test_real_frozen_long_context_gold_loads_both_splits_without_text_output(self):
        folder=Path(__file__).resolve().parents[1]/'evaluation/long-context-v1'
        for split in ('dev','heldout'):
            data=load_dataset(folder,split)
            mapping,digest=load_span_gold(data,split,folder)
            self.assertEqual(len(mapping),12)
            self.assertEqual(len(digest),64)
