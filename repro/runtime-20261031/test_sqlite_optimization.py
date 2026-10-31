import importlib.util,json,sqlite3,threading,time,uuid
from pathlib import Path
from unittest.mock import patch
from tests.common import MemoryTestCase
from tiger_memory import database,service

class SQLiteOptimizationTests(MemoryTestCase):
    def setUp(self):
        super().setUp();self.first=service.write_event(self.project,'first','Original cobalt',idempotency_key='first')
    def check_lookup(self,c):
        self.assertEqual(set(map(tuple,c.execute('SELECT rowid,record_type,ref_id FROM memory_fts'))),set(map(tuple,c.execute('SELECT fts_rowid,record_type,ref_id FROM fts_ref_lookup'))))
    def test_warm_connection_has_zero_writes_even_under_write_lock(self):
        holder=database.connect();holder.execute('BEGIN IMMEDIATE')
        started=time.monotonic();c=database.connect();elapsed=time.monotonic()-started
        self.assertEqual(c.total_changes,0);self.assertFalse(c.in_transaction);self.assertLess(elapsed,2);c.close();holder.rollback();holder.close()
    def test_reindex_delete_and_rollback_preserve_lookup(self):
        c=database.connect();self.check_lookup(c);c.execute('BEGIN IMMEDIATE');database.index_event(c,self.first);self.check_lookup(c);c.rollback();self.check_lookup(c)
        database._remove_search_record(c,'event',self.first.event_id);self.check_lookup(c);c.rollback();self.check_lookup(c);c.close()
    def test_migration_preserves_old_fts_rowids_and_duplicate_cleanup(self):
        c=database.connect();row=c.execute('SELECT ref_id,project_id,record_type,title,tags,body FROM memory_fts').fetchone()
        c.execute('INSERT INTO memory_fts(ref_id,project_id,record_type,title,tags,body) VALUES(?,?,?,?,?,?)',tuple(row));before=[tuple(x) for x in c.execute('SELECT rowid,* FROM memory_fts')]
        c.execute("DELETE FROM metadata WHERE key='write_optimization_version'");c.commit();c.close()
        c=database.connect();self.assertEqual(before,[tuple(x) for x in c.execute('SELECT rowid,* FROM memory_fts')]);self.check_lookup(c)
        database._remove_search_record(c,'event',self.first.event_id);self.assertEqual(c.execute('SELECT count(*) FROM memory_fts').fetchone()[0],0);self.check_lookup(c);c.rollback();c.close()
    def test_old_binary_writes_during_drain_keep_lookup_current(self):
        old=Path(__file__).resolve().parents[1]/'runtime-20261030/database.py'
        spec=importlib.util.spec_from_file_location('tiger_memory.legacy_write_test',old);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        c=m.connect();m._index_search_record(c,'event','old-writer','project','title','','legacy body');c.commit();self.check_lookup(c)
        m._index_search_record(c,'event','old-writer','project','title','','replacement body');c.commit();self.check_lookup(c)
        m._remove_search_record(c,'event','old-writer');c.commit();self.check_lookup(c);c.close()
    def test_vector_replacement_for_non_last_fts_row(self):
        c=database.connect();database._index_search_record(c,'event','later','project','title','','later');c.commit()
        r=c.execute('SELECT * FROM vectors WHERE ref_id=?',(self.first.event_id,)).fetchone();c.execute('INSERT OR REPLACE INTO vectors(record_type,ref_id,project_id,vector) VALUES(?,?,?,?)',tuple(r));c.commit();self.check_lookup(c);c.close()
    def test_busy_snapshot_and_other_errors_propagate(self):
        c=database.connect()
        for code in [517,sqlite3.SQLITE_LOCKED,sqlite3.SQLITE_IOERR,sqlite3.SQLITE_FULL,sqlite3.SQLITE_CORRUPT]:
            e=sqlite3.OperationalError('synthetic');e.sqlite_errorcode=code
            with patch.object(database.time,'sleep') as sleep:
                with self.assertRaises(sqlite3.OperationalError):c._busy_call(lambda:(_ for _ in ()).throw(e))
                sleep.assert_not_called()
        c.close()
    def test_busy_operation_not_transaction_replay(self):
        c=database.connect();calls=[]
        def operation():
            calls.append(1)
            if len(calls)<3:
                e=sqlite3.OperationalError('synthetic');e.sqlite_errorcode=sqlite3.SQLITE_BUSY;raise e
            return 'ok'
        with patch.object(database.time,'sleep'):self.assertEqual(c._busy_call(operation),'ok')
        self.assertEqual(len(calls),3);c.close()
    def test_busy_beyond_30_seconds_recovers_same_begin(self):
        holder=database.connect();holder.execute('BEGIN IMMEDIATE');started=threading.Event();result=[]
        def writer():
            c=database.connect();started.set();start=time.monotonic();c.execute('BEGIN IMMEDIATE');c.execute("INSERT INTO metadata(key,value) VALUES('synthetic_long_wait','once')");c.commit();result.append(time.monotonic()-start);c.close()
        t=threading.Thread(target=writer);t.start();self.assertTrue(started.wait(2));time.sleep(31.2);holder.commit();holder.close();t.join(10)
        self.assertFalse(t.is_alive());self.assertEqual(len(result),1);self.assertGreater(result[0],30)
        print(json.dumps({'sqlite_busy_wait_seconds':result[0],'same_operation_recovered':True}))
    def test_new_reference_cost_does_not_scan_all_fts(self):
        c=database.connect()
        for i in range(2000):database._index_search_record(c,'event','seed'+str(i),'p','t','','body')
        c.commit();steps=[0]
        def progress():steps[0]+=1;return 0
        c.set_progress_handler(progress,100)
        for i in range(10):database._index_search_record(c,'event','new'+str(i),'p','t','','body')
        c.set_progress_handler(None,0);c.commit();self.check_lookup(c)
        self.assertLess(steps[0],1000,steps)
        plan=c.execute('EXPLAIN QUERY PLAN DELETE FROM memory_fts WHERE rowid=? AND record_type=? AND ref_id=?',(1,'event','seed')).fetchall()
        print(json.dumps({'fts_rows':2011,'ten_inserts_sql_vm_steps_upper_bound':(steps[0]+1)*100,'rowid_cleanup_plan':[list(x) for x in plan]}));c.close()

    def test_interrupted_migration_rolls_back_then_recovers(self):
        c=database.connect();before=[tuple(r) for r in c.execute('SELECT rowid,* FROM memory_fts')]
        c.execute('DROP TRIGGER fts_ref_lookup_vectors_insert');c.execute('DROP TRIGGER fts_ref_lookup_vectors_delete');c.execute('DROP TABLE fts_ref_lookup');c.execute("DELETE FROM metadata WHERE key='write_optimization_version'");c.commit();c.close()
        execute=database.BusyRetryConnection.execute
        def interrupted(conn,sql,*args,**kwargs):
            if sql.startswith('CREATE TRIGGER'):raise OSError('synthetic migration interruption')
            return execute(conn,sql,*args,**kwargs)
        with patch.object(database.BusyRetryConnection,'execute',interrupted):
            with self.assertRaises(OSError):database.connect()
        c=database.connect_readonly();self.assertEqual(before,[tuple(r) for r in c.execute('SELECT rowid,* FROM memory_fts')]);self.assertIsNone(c.execute("SELECT name FROM sqlite_master WHERE name='fts_ref_lookup'").fetchone());c.close()
        c=database.connect();self.check_lookup(c);self.assertEqual(before,[tuple(r) for r in c.execute('SELECT rowid,* FROM memory_fts')]);c.close()
