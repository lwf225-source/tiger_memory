"""84 integrated HTTP -> source -> provider claims, synthetic transport only."""
import http.client
import json
import socket
import struct
import threading
import time
from unittest.mock import patch

from tiger_memory import aml_adapter as aml, request_budget as rb
from tests.test_aml_capacity import CapacityFixture
from tests.test_request_deadline import evidence


class RequestIntegrationTests(CapacityFixture):
    def setUp(self):
        self.records=[]
        make=aml.make_server
        with patch.object(aml,'make_server',side_effect=lambda host,port:make(host,port,
                request_timeout=1.2,max_connections=3,allow_test_deadline=True,
                request_observer=self.records.append)):
            super().setUp()

    def test_reserve_cancel_before_send_blocks_retry_and_preserves_source(self):
        checksum=self.budget._work_checksum
        def cancel(work):
            rb.current_request_budget().cancel('deadline')
            return checksum(work)
        with patch.object(self.budget,'_work_checksum',side_effect=cancel):
            self.assertEqual(self.add('r','A','s',['synthetic'])[0],503)
        first=self.sources();state=self.budget.snapshot()
        self.assertEqual((state['requests'],state['unknown'],state['claims']),(1,1,1))
        self.assertEqual(self.fake.requests,[])
        self.assertEqual(self.add('r','A','s',['changed ignored'])[0],503)
        self.assertEqual(self.sources(),first)
        self.assertEqual(self.add('new-r','A','s',['synthetic'])[0],503)
        self.assertEqual(self.budget.snapshot(),state)
        self.assertEqual(self.fake.requests,[])
        evidence('http-reserve-cancel',{'first_source_hashes':list(first.values()),'ledger':state,
                                      'same_id_and_changed_id_status':503,'sends':0})

    def test_late_valid_batch_keeps_raw_and_settle_then_retry_only_missing_new_work(self):
        original=self.fake.open
        def late(*args,**kwargs):
            time.sleep(rb.current_request_budget().remaining()+.04)
            return original(*args,**kwargs)
        self.fake.open=late
        self.assertEqual(self.add('r','A','s',[self.text(11)])[0],503)
        source=self.sources();state=self.budget.snapshot()
        self.assertEqual((state['settled'],state['claims']),(1,10))
        self.assertEqual(len(self.fake.requests),1)
        checkpoint=self.providers[0]._checkpoint.root
        self.assertEqual(len(list(checkpoint.glob('raw-*'))),10)
        self.assertEqual(len(list(checkpoint.glob('pooled-*'))),0)
        self.fake.open=original
        self.assertEqual(self.add('r','A','s',['changed ignored'])[0],200)
        self.assertEqual(self.sources(),source)
        self.assertEqual([len(x['input']) for x in self.fake.requests],[10,1])
        evidence('http-late-batch',{'source_unchanged':True,'batches':[10,1],
                                  'first_status':503,'retry_status':200,'ledger_after_late':state})

    def test_search_cold_then_warm_same_result_and_no_recharge(self):
        self.assertEqual(self.add('r','A','s',['violet evidence'])[0],200)
        first=self.search('violet','A',1)
        self.assertEqual(first[0],200)
        count=len(self.fake.requests);state=self.budget.snapshot()
        second=self.search('violet','A',1)
        self.assertEqual(second,first)
        self.assertEqual(len(self.fake.requests),count)
        self.assertEqual(self.budget.snapshot(),state)
        self.assertEqual(self.search('violet','unknown',1),(200,{'data':[]}))
        self.assertEqual(self.search('violet','A',0),(200,{'data':[]}))
        self.assertEqual(len(self.fake.requests),count)
        evidence('http-search-cache',{'normal_equal':True,'warm_extra_send':0,'unknown_topk0_extra_send':0})

    def test_rst_after_business_started_keeps_committed_work_without_refund(self):
        entered=threading.Event();release=threading.Event();original=self.fake.open
        def paused(*args,**kwargs):
            entered.set();release.wait(2);return original(*args,**kwargs)
        self.fake.open=paused
        payload={'request_id':'r','user_id':'A','session_id':'s',
                 'messages':[{'role':'user','content':'synthetic','timestamp':1704067200000}]}
        body=json.dumps(payload).encode()
        with socket.create_connection(('127.0.0.1',self.port),timeout=3) as sock:
            sock.sendall(b'POST /add HTTP/1.1\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            self.assertTrue(entered.wait(2))
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        release.set()
        end=time.monotonic()+2
        while not self.records and time.monotonic()<end:time.sleep(.01)
        self.assertTrue(self.records)
        self.assertEqual(self.records[0]['reason'],'disconnect')
        self.assertTrue(self.records[0]['business_completed'])
        self.assertFalse(self.records[0]['response_delivered'])
        self.assertEqual(self.budget.snapshot()['settled'],1)
        source=self.sources();self.fake.open=original
        self.assertEqual(self.request('POST','/add',payload=payload)[0],200)
        self.assertEqual(len(self.fake.requests),1);self.assertEqual(self.sources(),source)
        evidence('http-rst-after-start',{'record':self.records[0],'retry_send0':True,'source_preserved':True})
