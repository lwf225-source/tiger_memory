import copy,json,os,time,uuid
from pathlib import Path
from unittest.mock import patch
from tests.common import MemoryTestCase
from tiger_memory import service,database,reliability
from tiger_memory.events import serialize_event,event_relative_path
from tiger_memory.model import v2_root

class MetadataReuseTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        reliability._SOURCE_METADATA.clear()
        self.seed=service.write_event(self.project,'seed','ORIGINAL BODY',idempotency_key='seed')
        conn=database.connect_readonly()
        self.p=dict(conn.execute('SELECT * FROM projects WHERE project_id=?',(self.seed.project_id,)).fetchone());conn.close()
    def scan(self,key='absent',deleted=None):
        return reliability.find_orphan(self.p,key,service.batch_committed,deleted or set())
    def make(self,key,body='body',batch=None):
        m=copy.deepcopy(self.seed.metadata);m['id']=str(uuid.uuid4());m['idempotency_key']=key
        if batch:m['batch_id']=batch
        p=v2_root()/event_relative_path(self.p['folder_name'],m['id'],m['recorded_at'])
        p.write_text(serialize_event(m,body));return p,m
    def test_warm_miss_reads_no_bodies_but_new_orphan_is_seen(self):
        self.scan()
        with patch.object(reliability,'parse_event',wraps=reliability.parse_event) as parse:
            self.assertIsNone(self.scan());self.assertEqual(parse.call_count,0)
            path,m=self.make('crashed','FIRST BODY')
            self.assertEqual(self.scan('crashed').body,'FIRST BODY')
            self.assertEqual(parse.call_count,2)
    def test_same_size_edit_with_restored_mtime_is_detected(self):
        path,m=self.make('before');self.scan();info=path.stat()
        m['idempotency_key']='after!';path.write_text(serialize_event(m,'body'));os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns))
        self.assertIsNone(self.scan('before'));self.assertEqual(self.scan('after!').event_id,m['id'])
    def test_cached_body_is_never_returned(self):
        self.scan();self.seed.path.write_text(serialize_event(self.seed.metadata,'CURRENT BODY'))
        self.assertEqual(self.scan('seed').body,'CURRENT BODY')
    def test_delete_and_same_path_replacement(self):
        path,m=self.make('new');self.scan();path.unlink();self.assertIsNone(self.scan('new'))
        path.write_text(serialize_event(m,'REPLACEMENT'));self.assertEqual(self.scan('new').body,'REPLACEMENT')
    def test_corruption_and_duplicate_after_warmup(self):
        path,m=self.make('new');self.scan();m['idempotency_key']='seed';path.write_text(serialize_event(m,'duplicate'))
        with self.assertRaisesRegex(reliability.SourceRecoveryError,'AMBIGUOUS'):self.scan('seed')
        path.write_text('unknown corruption')
        with self.assertRaisesRegex(reliability.SourceRecoveryError,'CORRUPT'):self.scan()
    def test_symlink_parent_after_warmup(self):
        self.scan();parent=self.seed.path.parent;moved=parent.with_name(parent.name+'-moved');parent.rename(moved);parent.symlink_to(moved,target_is_directory=True)
        with self.assertRaisesRegex(reliability.SourceRecoveryError,'CORRUPT'):self.scan()
    def test_symlink_file_after_warmup(self):
        self.scan();p=self.seed.path;q=p.with_suffix('.bak');p.rename(q);p.symlink_to(q)
        with self.assertRaisesRegex(reliability.SourceRecoveryError,'CORRUPT'):self.scan()
    def test_batch_and_deletion_decisions_not_cached(self):
        batch=str(uuid.uuid4());path,m=self.make('batch',batch=batch);self.assertIsNone(self.scan('batch'))
        marker=v2_root()/'batches'/(batch+'.json');marker.parent.mkdir(exist_ok=True);marker.write_text(json.dumps({'project_id':m['project_id'],'event_ids':[m['id']]}))
        self.assertEqual(self.scan('batch').event_id,m['id']);self.assertIsNone(self.scan('batch',{m['id']}));marker.unlink();self.assertIsNone(self.scan('batch'))
    def test_eviction_only_forces_revalidation(self):
        self.make('one');self.make('two')
        with patch.object(reliability,'_SOURCE_METADATA_LIMIT',1):
            self.scan();self.assertLessEqual(len(reliability._SOURCE_METADATA),1);self.assertEqual(self.scan('seed').body,'ORIGINAL BODY')
    def test_idempotency_exact_partial_index(self):
        conn=database.connect_readonly();queries=[];conn.set_trace_callback(queries.append)
        self.assertIsNotNone(database.event_for_idempotency(conn,self.seed.project_id,'seed'))
        plan=conn.execute('EXPLAIN QUERY PLAN '+queries[-1]).fetchall()
        self.assertTrue(any('events_idempotency' in str(tuple(r)) for r in plan),plan);conn.close()
    def test_500_file_read_amplification(self):
        for i in range(500):self.make('history-'+str(i),'long history '*800)
        original=reliability.parse_event
        with patch.object(reliability,'parse_event',wraps=original) as parse:
            started=time.monotonic();self.scan();cold=time.monotonic()-started;reads=parse.call_count
            parse.reset_mock();started=time.monotonic()
            for _ in range(10):self.scan()
            warm=(time.monotonic()-started)/10
            self.assertEqual(parse.call_count,0)
        print(json.dumps({'benchmark':'501_source_files','cold_seconds':cold,'cold_parses':reads,'warm_seconds_per_scan':warm,'warm_body_parses':0,'speedup':cold/warm}))
