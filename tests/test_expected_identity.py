"""Fixed storage identity: synthetic stores, fake providers, real local crashes."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from unittest.mock import patch

from tests.common import MemoryTestCase
from tests.test_aml_adapter import AMLServerCase
from tiger_memory import aml_adapter as aml, database, events, model, service
from tiger_memory.reliability import SourceRecoveryError


class FixedIdentityTests(MemoryTestCase):
    def setUp(self):
        super().setUp()
        self.env = patch.dict(os.environ, {'HOME': str(Path(self.temp.name)/'home'),
            'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.a = aml.ensure_user_storage('A')
        self.b = aml.ensure_user_storage('B')
        self.identity = service.ExpectedIdentity(self.a, 'session', 'A')

    def write(self, body='synthetic A canary', key='aml:A:r:0', **kwargs):
        return service.write_event(self.a, 'original title', body, scope='session', scope_id='A',
            idempotency_key=key, expected_identity=self.identity, **kwargs)

    def rows(self, table='events'):
        if not database.db_path().exists(): return []
        conn=database.connect_readonly()
        try:return [dict(row) for row in conn.execute('SELECT * FROM '+table)]
        finally:conn.close()

    def hashes(self):
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.base.glob('v2/projects/*/events/*/*/*.md')}

    def alias_to_b(self):
        service.write_event(self.b,'B title','synthetic B canary',scope='session',scope_id='B')
        conn=database.connect()
        conn.execute('INSERT OR REPLACE INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?)',
                     (str(self.a),model.project_id_for_root(self.b),model.utc_now()))
        conn.commit();conn.close()

    def git_init(self):
        subprocess.run(['git','init','-q',str(self.base)],check=True,capture_output=True)

    def orphan(self, scope_id='A'):
        original=service.atomic_write
        def failed(path,text):
            original(path,text)
            if path.suffix=='.md':raise OSError('synthetic failure after source publication')
        with patch.object(service,'atomic_write',failed),self.assertRaises(OSError):
            service.write_event(self.a,'orphan title','synthetic original orphan',scope='session',
                                scope_id=scope_id,idempotency_key='aml:A:r:0',
                                expected_identity=self.identity if scope_id=='A' else None)
        return events.parse_event(Path(next(iter(self.hashes()))).resolve())

    def test_normal_first_write_and_same_text_user_isolation(self):
        first=self.write('shared synthetic text')
        again=self.write('changed replay')
        self.assertEqual(first,again)
        other=service.write_event(self.b,'same title','shared synthetic text',scope='session',scope_id='B',
            idempotency_key='aml:B:r:0',expected_identity=service.ExpectedIdentity(self.b,'session','B'))
        self.assertNotEqual(first.project_id,other.project_id)
        self.assertEqual(len(self.rows()),2)
        self.assertEqual(self.identity.project_id,model.project_id_for_root(self.a))
        self.assertEqual(service._expected_project(self.identity)['folder_name'],model.display_slug(self.a))

    def test_fixed_folder_formula_matches_native_without_reresolution(self):
        for root in [Path('/'),self.a,Path(self.temp.name)/'中文 dotted.name']:
            identity=service.ExpectedIdentity(root.resolve(),'session','A')
            self.assertEqual(service._expected_project(identity)['folder_name'],model.display_slug(root))

    def test_alias_conflict_before_registration_and_after_adapter_preflight(self):
        aml._storage_identity_preflight('A')
        self.alias_to_b();before=self.hashes();rows=self.rows()
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        self.assertEqual(before,self.hashes());self.assertEqual(rows,self.rows())
        self.assertFalse(any(p['project_id']==self.identity.project_id for p in self.rows('projects')))

    def test_alias_change_between_outer_check_and_sql_transaction(self):
        original=service._validate_expected_project
        changed=False
        def injected(conn,identity):
            nonlocal changed
            result=original(conn,identity)
            if conn is None and not changed:
                changed=True;self.alias_to_b()
            return result
        with patch.object(service,'_validate_expected_project',injected),self.assertRaises(service.StorageIdentityConflict):
            self.write()
        self.assertEqual([r['scope_id'] for r in self.rows()],['B'])

    def test_git_after_adapter_preflight_rejects_before_registration(self):
        aml._storage_identity_preflight('A');self.git_init()
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        self.assertEqual(self.hashes(),{});self.assertEqual(self.rows(),[])
        self.assertEqual(self.rows('projects'),[])

    def test_git_change_after_index_before_publication_rolls_back(self):
        original=service._index_validated_write
        def injected(*a,**kw):original(*a,**kw);self.git_init()
        with patch.object(service,'_index_validated_write',injected),self.assertRaises(service.StorageIdentityConflict):
            self.write()
        self.assertEqual(self.hashes(),{});self.assertEqual(self.rows(),[])
        self.assertEqual(self.rows('projects'),[])

    def test_alias_change_inside_transaction_before_publication_rolls_back(self):
        service.write_event(self.b,'B','synthetic B',scope='session',scope_id='B')
        original=service._index_validated_write;before=self.hashes()
        def injected(conn,*a,**kw):
            original(conn,*a,**kw)
            conn.execute('UPDATE project_aliases SET project_id=? WHERE alias_path=?',
                         (model.project_id_for_root(self.b),str(self.a)))
        with patch.object(service,'_index_validated_write',injected),self.assertRaises(service.StorageIdentityConflict):
            self.write()
        self.assertEqual(self.hashes(),before)
        self.assertEqual([r['scope_id'] for r in self.rows()],['B'])

    def test_last_observation_window_keeps_fixed_target(self):
        original=service.atomic_write
        def injected(path,text):
            if path.suffix=='.md':self.git_init()
            return original(path,text)
        with patch.object(service,'atomic_write',injected):event=self.write()
        self.assertEqual(event.project_id,self.identity.project_id)
        self.assertEqual(event.metadata['project_root'],str(self.a))
        self.assertEqual(self.rows()[0]['project_id'],self.identity.project_id)
        self.assertNotEqual(event.project_id,model.project_id_for_root(self.base))
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        shutil.rmtree(self.base/'.git')
        self.assertEqual(self.write().event_id,event.event_id)

    def test_foreign_orphan_rejected_without_import(self):
        self.orphan('B');before=self.hashes()
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        self.assertEqual(self.hashes(),before);self.assertEqual(self.rows(),[])

    def test_ambiguous_orphans_still_raise_recovery_error(self):
        orphan=self.orphan();meta=dict(orphan.metadata,id=str(uuid.uuid4()))
        path=model.v2_root()/events.event_relative_path(model.display_slug(self.a),meta['id'],meta['recorded_at'])
        events.atomic_write(path,events.serialize_event(meta,orphan.body));before=self.hashes()
        with self.assertRaisesRegex(SourceRecoveryError,'SOURCE_RECOVERY_AMBIGUOUS'):self.write()
        self.assertEqual(self.hashes(),before);self.assertEqual(self.rows(),[])

    def test_legal_orphan_recovery_and_replay_preserve_source(self):
        orphan=self.orphan();before=self.hashes()
        recovered=self.write('changed body')
        self.assertEqual(recovered,orphan)
        self.assertEqual(self.hashes(),before)
        self.assertEqual(self.write().event_id,orphan.event_id)
        self.assertEqual(len(self.rows()),1)

    def test_idempotent_foreign_scope_rejected_not_returned(self):
        service.write_event(self.a,'foreign','synthetic B scope',scope='session',scope_id='B',idempotency_key='aml:A:r:0')
        before=self.hashes();rows=self.rows()
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        self.assertEqual(before,self.hashes());self.assertEqual(rows,self.rows())

    def test_idempotent_index_scope_mismatch_rejected(self):
        self.write();before=self.hashes()
        conn=database.connect();conn.execute("UPDATE events SET scope_id='B'");conn.commit();conn.close()
        with self.assertRaises(service.StorageIdentityConflict):self.write()
        self.assertEqual(before,self.hashes())

    def test_identity_scope_arguments_and_existing_batch_fail_closed(self):
        with self.assertRaises(service.StorageIdentityConflict):
            service.write_event(self.a,'title','synthetic',scope='session',scope_id='B',expected_identity=self.identity)
        with service.event_batch(self.a,'synthetic-batch'):
            with self.assertRaisesRegex(model.PLMError,'cannot use event_batch'):self.write()
        self.assertEqual(self.rows(),[])

    def test_search_alias_conflict_precedes_provider_and_precontamination(self):
        self.write();self.alias_to_b()
        service.write_event(self.b,'foreign','synthetic A scope in B',scope='session',scope_id='A')
        with patch.object(service,'search',side_effect=AssertionError('retrieval must not run')):
            with self.assertRaises(service.StorageIdentityConflict):
                service.context_data(self.a,'synthetic',scope='session',scope_id='A',expected_identity=self.identity)

    def test_search_git_conflict_precedes_provider(self):
        self.write();self.git_init()
        with patch.object(service,'search',side_effect=AssertionError('retrieval must not run')):
            with self.assertRaises(service.StorageIdentityConflict):
                service.context_data(self.a,'synthetic',scope='session',scope_id='A',expected_identity=self.identity)

    def test_search_late_alias_change_stays_on_captured_project(self):
        event=self.write()
        calls=[]
        class Fake:
            def describe(self):return {'status':'ready'}
            def embed(inner,texts):calls.append(list(texts));return [[1.,0.] for _ in texts]
        original=service.search
        def injected(*a,**kw):self.alias_to_b();return original(*a,**kw)
        with patch.object(service,'search',injected):
            project,results=service.context_data(self.a,'synthetic',scope='session',scope_id='A',
                expected_identity=self.identity,embedding_provider=Fake())
        self.assertEqual(project['project_id'],event.project_id)
        self.assertTrue(results);self.assertTrue(calls)
        self.assertTrue(all(r.project_id==event.project_id for r in results))
        self.assertFalse(any('synthetic B canary' in text for call in calls for text in call))

    def test_search_does_not_hold_store_lock_across_provider(self):
        self.write()
        test=self
        class Fake:
            def describe(self):return {'status':'ready'}
            def embed(inner,texts):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future=pool.submit(test.write,'synthetic parallel','aml:A:parallel:0')
                    future.result(timeout=5)
                return [[1.,0.] for _ in texts]
        service.context_data(self.a,'synthetic',scope='session',scope_id='A',
            expected_identity=self.identity,embedding_provider=Fake())
        self.assertEqual(len(self.rows()),2)

    def test_search_filters_other_project_and_scope_before_fake_embedding(self):
        first=self.write('synthetic legitimate A only')
        service.write_event(self.a,'wrong scope','synthetic foreign scope canary',scope='session',scope_id='B')
        service.write_event(self.b,'wrong project','synthetic foreign project canary',scope='session',scope_id='A')
        calls=[]
        class Fake:
            def describe(self):return {'status':'ready'}
            def embed(inner,texts):calls.extend(texts);return [[1.,0.] for _ in texts]
        project,results=service.context_data(self.a,'synthetic',scope='session',scope_id='A',
            expected_identity=self.identity,embedding_provider=Fake())
        self.assertEqual(project['project_id'],first.project_id)
        self.assertEqual([r.ref_id for r in results],[first.event_id])
        self.assertFalse(any('foreign' in text for text in calls))

    def test_indexed_foreign_source_path_rejected_before_file_read(self):
        self.write()
        conn=database.connect()
        conn.execute('UPDATE events SET source_path=?',(str(Path(self.temp.name)/'never-read.md'),))
        conn.commit();conn.close()
        with patch.object(service,'parse_event',side_effect=AssertionError('foreign source must not be read')):
            with self.assertRaises(service.StorageIdentityConflict):self.write()
            with self.assertRaises(service.StorageIdentityConflict):
                service.preview_event_for_idempotency(self.a,'aml:A:r:0',expected_identity=self.identity)

    def test_preview_existing_and_orphan_never_imports(self):
        self.assertIsNone(service.preview_event_for_idempotency(self.a,'aml:A:r:0',expected_identity=self.identity))
        self.assertFalse(database.db_path().exists())
        orphan=self.orphan();before=self.hashes()
        found=service.preview_event_for_idempotency(self.a,'aml:A:r:0',expected_identity=self.identity)
        self.assertEqual(found,orphan);self.assertEqual(self.rows(),[]);self.assertEqual(before,self.hashes())
        self.write()
        self.assertEqual(service.preview_event_for_idempotency(self.a,'aml:A:r:0',expected_identity=self.identity),orphan)

    def test_preview_foreign_orphan_fails_without_import(self):
        self.orphan('B');before=self.hashes()
        with self.assertRaises(service.StorageIdentityConflict):
            service.preview_event_for_idempotency(self.a,'aml:A:r:0',expected_identity=self.identity)
        self.assertEqual(before,self.hashes());self.assertEqual(self.rows(),[])

    def test_cli_alias_preserved_without_expected_identity(self):
        first=service.write_event(self.project,'old','synthetic cli')
        moved=Path(self.temp.name)/'moved';moved.mkdir();(moved/'pyproject.toml').touch()
        service.add_project_alias(self.project,moved)
        second=service.write_event(moved,'moved','synthetic cli continued')
        self.assertEqual(first.project_id,second.project_id)

    def test_concurrent_same_key_returns_single_event(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(self.write,'synthetic concurrent') for _ in range(8)]
            ids=[f.result(timeout=15).event_id for f in futures]
        self.assertEqual(len(set(ids)),1);self.assertEqual(len(self.hashes()),1);self.assertEqual(len(self.rows()),1)

    def child(self, crash=False):
        code='''import json,os,signal
from pathlib import Path
from tiger_memory import service
root=Path(os.environ['SYNTHETIC_ROOT'])
original=service.atomic_write
def publish(path,text):
    original(path,text)
    if path.suffix=='.md':os.kill(os.getpid(),signal.SIGKILL)
if os.environ['SYNTHETIC_CRASH']=='1':service.atomic_write=publish
event=service.write_event(root,'child original','synthetic child source',scope='session',scope_id='A',
    idempotency_key='aml:A:r:0',expected_identity=service.ExpectedIdentity(root,'session','A'))
print(json.dumps({'id':event.event_id,'body':event.body}))
'''
        env={'PATH':'/usr/bin:/bin','HOME':str(Path(self.temp.name)/'child-home'),
             'TIGER_MEMORY_DIR':str(self.base),'SYNTHETIC_ROOT':str(self.a),'SYNTHETIC_CRASH':str(int(crash)),
             'PYTHONPATH':str(Path(service.__file__).resolve().parent.parent),
             'PYTHONDONTWRITEBYTECODE':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
        return subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

    def test_sigkill_after_source_publication_new_process_recovers(self):
        crashed=self.child(True);out,err=crashed.communicate(timeout=15)
        self.assertEqual(crashed.returncode,-9,(out,err));before=self.hashes()
        self.assertEqual(len(before),1);self.assertEqual(self.rows(),[])
        restarted=self.child();out,err=restarted.communicate(timeout=15)
        self.assertEqual(restarted.returncode,0,(out,err));recovered=json.loads(out)
        self.assertEqual(before,self.hashes());self.assertEqual(self.rows()[0]['event_id'],recovered['id'])

    def test_two_processes_same_key_no_duplicate_source(self):
        children=[self.child(),self.child()]
        results=[]
        for child in children:
            out,err=child.communicate(timeout=15);self.assertEqual(child.returncode,0,(out,err));results.append(json.loads(out))
        self.assertEqual(results[0],results[1]);self.assertEqual(len(self.hashes()),1);self.assertEqual(len(self.rows()),1)


class FixedIdentityHTTPTests(AMLServerCase):
    """Integration seam: run only after the entry owner's identity wiring."""
    def setUp(self):
        super().setUp()
        self.sent=[]
        test=self
        class Fake:
            def plan(self,texts):return {'within_request_batch_limit':True}
            def describe(self):return {'status':'ready'}
            def embed(self,texts):test.sent.append(list(texts));return [[1.,0.] for _ in texts]
        settings={'AML_EMBEDDING':'remote-v4','AML_STRICT_RETRIEVAL':True,
                  'AML_EMBEDDING_FACTORY':lambda user:Fake(),'AML_FACT_PROJECTION':'off',
                  'AML_QUERY_REWRITE':'off','AML_RERANKER':'off'}
        self.patches=[patch.object(aml,k,v) for k,v in settings.items()]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        env=patch.dict(os.environ,{'AML_API_KEY':'','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'})
        env.start();self.addCleanup(env.stop)

    def sources(self):
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.base.glob('v2/projects/*/events/*/*/*.md')}

    def rows(self):
        if not database.db_path().exists():return []
        c=database.connect_readonly()
        try:return [dict(r) for r in c.execute('SELECT * FROM events')]
        finally:c.close()

    def preflight_mutation(self, mutation):
        original=aml._storage_identity_preflight
        def injected(user):
            root=original(user);mutation(root);return root
        return patch.object(aml,'_storage_identity_preflight',injected)

    def alias_to_b(self,root):
        conn=database.connect()
        conn.execute('INSERT OR REPLACE INTO project_aliases(alias_path,project_id,active,created_at) VALUES(?,?,1,?)',
                     (str(root),model.project_id_for_root(aml.user_storage_dir('B')),model.utc_now()))
        conn.commit();conn.close()

    def test_add_alias_after_preflight_503_no_wrong_source_or_embedding(self):
        self.assertEqual(self.add('r','B','s',['synthetic B'])[0],200)
        before=self.sources();rows=self.rows();self.sent.clear()
        with self.preflight_mutation(self.alias_to_b):status,body=self.add('r','A','s',['synthetic A'])
        self.assertEqual(status,503);self.assertEqual(body['detail']['reason'],'storage_identity_conflict')
        self.assertEqual(before,self.sources());self.assertEqual(rows,self.rows());self.assertEqual(self.sent,[])

    def test_add_git_after_preflight_503_no_source_or_embedding(self):
        def change(root):subprocess.run(['git','init','-q',str(self.base)],check=True,capture_output=True)
        with self.preflight_mutation(change):status,body=self.add('r','A','s',['synthetic A'])
        self.assertEqual(status,503);self.assertEqual(body['detail']['reason'],'storage_identity_conflict')
        self.assertEqual(self.rows(),[]);self.assertEqual(self.sources(),{});self.assertEqual(self.sent,[])

    def test_foreign_orphan_503_no_import_or_embedding(self):
        root=aml.ensure_user_storage('A')
        event=service.write_event(root,'orphan','synthetic foreign',scope='session',scope_id='B',idempotency_key='aml:A:r:0')
        conn=database.connect()
        for table,key in [('events','event_id'),('memory_fts','ref_id'),('vectors','ref_id')]:
            conn.execute('DELETE FROM '+table+' WHERE '+key+'=?',(event.event_id,))
        conn.commit();conn.close();before=self.sources()
        status,body=self.add('r','A','s',['synthetic A'])
        self.assertEqual(status,503);self.assertEqual(body['detail']['reason'],'storage_identity_conflict')
        self.assertEqual(before,self.sources());self.assertEqual(self.rows(),[]);self.assertEqual(self.sent,[])

    def test_search_precontaminated_alias_503_before_embedding(self):
        self.assertEqual(self.add('r','A','s',['synthetic A'])[0],200)
        self.assertEqual(self.add('r','B','s',['synthetic B'])[0],200)
        service.write_event(aml.user_storage_dir('B'),'foreign','synthetic polluted A scope',scope='session',scope_id='A')
        self.sent.clear()
        with self.preflight_mutation(self.alias_to_b):status,body=self.search('synthetic','A')
        self.assertEqual(status,503);self.assertEqual(body['detail']['reason'],'storage_identity_conflict')
        self.assertEqual(self.sent,[])

    def test_partial_prefix_and_corrected_topology_retry_remain_recoverable(self):
        original=service._index_validated_write;calls=0
        def injected(*a,**kw):
            nonlocal calls
            original(*a,**kw);calls+=1
            if calls==2:subprocess.run(['git','init','-q',str(self.base)],check=True,capture_output=True)
        with patch.object(service,'_index_validated_write',injected):
            status,_=self.add('r','A','s',['synthetic first','synthetic second'])
        self.assertEqual(status,503);self.assertEqual(len(self.rows()),1);self.assertEqual(self.sent,[])
        first=self.rows()[0]['event_id'];shutil.rmtree(self.base/'.git')
        self.assertEqual(self.add('r','A','s',['synthetic first','synthetic second'])[0],200)
        self.assertEqual(len(self.rows()),2);self.assertIn(first,[r['event_id'] for r in self.rows()])

    def test_normal_http_same_text_and_canary_isolation(self):
        for user in ['A','B']:
            self.assertEqual(self.add('r',user,'s',['shared synthetic',user+' private canary'])[0],200)
        before=self.sources()
        self.assertEqual(self.add('r','A','s',['changed retry','changed canary'])[0],200)
        self.assertEqual(before,self.sources())
        for user,other in [('A','B'),('B','A')]:
            status,body=self.search('shared canary',user)
            self.assertEqual(status,200)
            contents=[r['content'] for r in body['data']]
            self.assertIn('shared synthetic',contents);self.assertIn(user+' private canary',contents)
            self.assertNotIn(other+' private canary',contents)
