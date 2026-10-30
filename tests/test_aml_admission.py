"""Legitimate concurrent remote Add serialization and crash-released lock."""
import concurrent.futures
import os
from pathlib import Path
import signal
import subprocess
from tests.child_env import child_popen
import sys
import threading
import time
from unittest.mock import patch
from tests.test_aml_capacity import CapacityFixture
from tests.test_embedding_budget import evidence
from tiger_memory import aml_adapter as aml
from tiger_memory.aml_add_lock import add_admission_lock

class AddAdmissionTests(CapacityFixture):
    maxDiff = None
    def test_changed_payload_same_key_requests_do_not_mix_first_write(self):
        big_a = "a" * (9*8192 - 100)
        big_b = "b" * (9*8192 - 100)
        first_written = threading.Event()
        release = threading.Event()
        b_entered_write = threading.Event()
        original_write = aml.service.write_event
        original_factory = aml.AML_EMBEDDING_FACTORY
        def factory(user):
            obj = original_factory(user)
            obj.max_request_batches = 1
            return obj
        def write(root,title,content,**kwargs):
            if content == "small-b":
                b_entered_write.set()
            event = original_write(root,title,content,**kwargs)
            if content == big_a:
                first_written.set()
                assert release.wait(5)
            return event
        with patch.object(aml,"AML_EMBEDDING_FACTORY",side_effect=factory), patch.object(aml.service,"write_event",side_effect=write):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(self.add,"same-id","A","s",[big_a,"small-a"])
                self.assertTrue(first_written.wait(5))
                b = pool.submit(self.add,"same-id","A","s",["small-b",big_b])
                self.assertFalse(b_entered_write.wait(.15))
                release.set()
                self.assertEqual([a.result(10)[0],b.result(10)[0]],[200,200])
        self.assertEqual(len(self.sources()),2)
        self.assertEqual({p[1]["total_chunks"] for p in self.plans},{10})
        self.assertTrue(all(text.split("\n",2)[-1] in {big_a,"small-a"} for texts,_ in self.plans for text in texts))
        self.assertEqual(sum(len(request["input"]) for request in self.fake.requests),10)
        self.assertEqual(self.add("same-id","A","s",["small-b",big_b])[0],200)
        self.assertEqual(sum(len(request["input"]) for request in self.fake.requests),10)

    def test_bounded_lock_wait_returns_503_without_source_or_send_then_recovers(self):
        root = aml.ensure_user_storage("A").resolve()
        with add_admission_lock(root), patch.object(aml,"AML_ADD_LOCK_TIMEOUT_SECONDS",.05):
            status,body = self.add("r","A","s",["original"])
        self.assertEqual((status,body["detail"]["reason"]),(503,"add temporarily busy"))
        self.assertEqual(self.sources(),{})
        self.assertEqual(self.fake.requests,[])
        self.assertEqual(self.add("r","A","s",["original"])[0],200)

    def test_different_users_and_remote_calls_are_outside_admission_lock(self):
        root = aml.ensure_user_storage("A").resolve()
        with add_admission_lock(root), patch.object(aml,"AML_ADD_LOCK_TIMEOUT_SECONDS",.05):
            self.assertEqual(self.add("r","B","s",["B original"])[0],200)
        original = aml.AML_EMBEDDING_FACTORY
        def factory(user):
            obj=original(user)
            remote=obj.embed
            def embed(texts):
                with add_admission_lock(aml.user_storage_dir(user).resolve(),.05):
                    pass
                return remote(texts)
            obj.embed=embed
            return obj
        with patch.object(aml,"AML_EMBEDDING_FACTORY",side_effect=factory):
            self.assertEqual(self.add("r","A","s",["A original"])[0],200)

    def test_independent_process_lock_blocks_until_sigkill_then_retry_succeeds(self):
        root=aml.ensure_user_storage("A").resolve()
        ready=Path(self.temp.name)/"ready"
        # Synthetic child imports the same source/install package via parent cwd.
        code="""import sys,time
from pathlib import Path
from tiger_memory.aml_add_lock import add_admission_lock
with add_admission_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready')
    time.sleep(30)
"""
        child=child_popen([sys.executable,"-B","-c",code,str(root),str(ready)],
            env={"PATH":os.defpath,"HOME":str(Path(self.temp.name)/"home"),"PYTHONDONTWRITEBYTECODE":"1"},
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            end=time.monotonic()+5
            while not ready.exists() and child.poll() is None and time.monotonic()<end:
                time.sleep(.02)
            self.assertTrue(ready.exists())
            with patch.object(aml,"AML_ADD_LOCK_TIMEOUT_SECONDS",.05):
                self.assertEqual(self.add("r","A","s",["original"])[0],503)
            self.assertEqual(self.sources(),{})
            self.assertEqual(self.fake.requests,[])
            child.kill()
            self.assertEqual(child.wait(5),-signal.SIGKILL)
            self.assertEqual(self.add("r","A","s",["original"])[0],200)
        finally:
            if child.poll() is None:child.kill()
            stdout,stderr=child.communicate(timeout=5)
            self.assertEqual((stdout,stderr),(b"",b""))

    def _check_crash_recovery(self, committed):
        ready=Path(self.temp.name)/"source-ready"
        code="""import os,sys,time
from pathlib import Path
from tiger_memory import aml_adapter as aml,service
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event in ('socket.connect','socket.getaddrinfo') else None)
os.environ['TIGER_MEMORY_DIR']=sys.argv[1]
for name,value in {'AML_EMBEDDING':'remote-v4','AML_STRICT_RETRIEVAL':True,'AML_RERANKER':'off','AML_QUERY_REWRITE':'off','AML_FACT_PROJECTION':'off','AML_RECENCY_BOOST':False,'AML_NARRATIVE_TIME':False,'AML_SESSION_RRF':False,'AML_SESSION_VIEWS':False}.items():setattr(aml,name,value)
aml.AML_EMBEDDING_FACTORY=lambda user:DashScopeEmbeddingProvider('http://127.0.0.1:9/embeddings','SYNTHETIC_ONLY',dimensions=64,allow_test_loopback=True,max_request_batches=1)
original=service.atomic_write
def write(path,text):
    original(path,text)
    if path.suffix=='.md':
        Path(sys.argv[2]).write_text('source-published')
        time.sleep(30)
service.atomic_write=write
aml.add({'request_id':'r','user_id':'A','session_id':'s','messages':[{'role':'user','content':'a'*(9*8192-100),'timestamp':1704067200000},{'role':'assistant','content':'small-a','timestamp':1704067260000}]})
"""
        if committed:
            old = """original=service.atomic_write
def write(path,text):
    original(path,text)
    if path.suffix=='.md':
        Path(sys.argv[2]).write_text('source-published')
        time.sleep(30)
service.atomic_write=write"""
            new = """original=service.write_event
def write(*args,**kwargs):
    event=original(*args,**kwargs)
    Path(sys.argv[2]).write_text('source-committed')
    time.sleep(30)
    return event
service.write_event=write"""
            assert old in code
            code=code.replace(old,new)
        child=child_popen([sys.executable,"-B","-c",code,str(self.base.resolve()),str(ready)],
            env={"PATH":os.defpath,"HOME":str(Path(self.temp.name)/"home"),"PYTHONDONTWRITEBYTECODE":"1",
                 "GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            end=time.monotonic()+10
            while not ready.exists() and child.poll() is None and time.monotonic()<end:
                time.sleep(.02)
            self.assertTrue(ready.exists())
            first_source=self.sources()
            self.assertEqual(len(first_source),1)
            child.kill()
            self.assertEqual(child.wait(5),-signal.SIGKILL)
            original=aml.AML_EMBEDDING_FACTORY
            def factory(user):
                obj=original(user)
                obj.max_request_batches=1
                return obj
            with patch.object(aml,"AML_EMBEDDING_FACTORY",side_effect=factory):
                status,body=self.add("r","A","s",["small-b","b"*(9*8192-100)])
                self.assertEqual((status,body["detail"]["reason"]),(503,"embedding request capacity exceeded"))
                self.assertEqual(self.plans[-1][1]["total_chunks"],18)
                self.assertEqual(self.sources(),first_source)
                self.assertEqual(self.fake.requests,[])
                self.assertEqual(self.add("r","A","s",["a"*(9*8192-100),"small-a"])[0],200)
            self.assertEqual(self.plans[-1][1]["total_chunks"],10)
            self.assertEqual(len(self.sources()),2)
            self.assertTrue(first_source.items() <= self.sources().items())
            self.assertEqual(sum(len(p["input"]) for p in self.fake.requests),10)
            self.assertEqual(self.budget.snapshot()["requests"],1)
            evidence("admission-crash-" + ("committed" if committed else "orphan"), {
                "child_exit": -signal.SIGKILL, "competing_plan_chunks": 18, "competing_status": 503,
                "competing_second_source_added": 0, "competing_sends": 0, "original_plan_chunks": 10,
                "original_status": 200, "final_sources": 2, "original_hash_preserved": True,
                "fake_rows": 10, "budget_requests": 1})
        finally:
            if child.poll() is None:child.kill()
            stdout,stderr=child.communicate(timeout=5)
            self.assertEqual((stdout,stderr),(b"",b""))

    def test_two_independent_http_processes_first_write_and_capacity_stay_together(self):
        import http.client
        import json
        root=Path(self.temp.name).resolve()
        receipt=root/"receipt.json"
        receipt.write_text(json.dumps(self.budget.receipt()))
        receipt.chmod(0o600)
        env={"PATH":os.defpath,"HOME":str(root/"home"),"PYTHONDONTWRITEBYTECODE":"1",
            "PYTHONPATH":str(Path(__file__).resolve().parents[1]),"TIGER_MEMORY_DIR":str(self.base.resolve()),
            "GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull}
        children=[]
        def start(role):
            child=child_popen([sys.executable,"-B","-m","tests.aml_admission_worker",str(root),role],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            children.append(child)
            end=time.monotonic()+10
            while not (root/("port-"+role)).exists() and child.poll() is None and time.monotonic()<end:time.sleep(.02)
            self.assertTrue((root/("port-"+role)).exists())
            return int((root/("port-"+role)).read_text())
        def request(port,texts):
            conn=http.client.HTTPConnection("127.0.0.1",port,timeout=15)
            try:
                conn.request("POST","/add",json.dumps({"request_id":"r","user_id":"A","session_id":"s",
                    "messages":[{"role":"user" if i==0 else "assistant","content":text,"timestamp":1704067200000+i*60000} for i,text in enumerate(texts)]}))
                response=conn.getresponse();response.read();return response.status
            finally:conn.close()
        try:
            ports=[start(role) for role in ("A","B")]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a=pool.submit(request,ports[0],["a"*(9*8192-100),"small-a"])
                end=time.monotonic()+5
                while not (root/"first-committed").exists() and time.monotonic()<end:time.sleep(.01)
                self.assertTrue((root/"first-committed").exists())
                b=pool.submit(request,ports[1],["small-b","b"*(9*8192-100)])
                time.sleep(.15)
                self.assertFalse((root/"plan-B").exists())
                (root/"release").write_text("continue")
                self.assertEqual([a.result(10),b.result(10)],[200,200])
            plans=[json.loads((root/("plan-"+role)).read_text()) for role in ("A","B")]
            self.assertEqual([p["chunks"] for p in plans],[10,10])
            self.assertEqual(plans[0]["lock_inode"],plans[1]["lock_inode"])
            self.assertEqual(len(self.sources()),2)
            self.assertEqual((root/"wire.log").read_text().splitlines(),["10"])
            self.assertEqual(self.budget.snapshot()["requests"],1)
            evidence("admission-two-http-processes", {"statuses": [200, 200], "plan_chunks": [10, 10],
                "same_lock_inode": True, "final_sources": 2, "fake_batches": [10], "budget_requests": 1})
        finally:
            (root/"release").write_text("continue")
            for child in children:
                if child.poll() is None:child.kill()
                stdout,stderr=child.communicate(timeout=5)
                self.assertEqual((stdout,stderr),(b"",b""))

    def test_kill_after_orphan_publication_competitor_rejects_and_original_recovers(self):
        self._check_crash_recovery(False)

    def test_kill_after_first_commit_competitor_rejects_and_original_recovers(self):
        self._check_crash_recovery(True)

    def test_blocked_embed_allows_same_user_next_source_phase(self):
        embedding=threading.Event();release=threading.Event();second_written=threading.Event()
        original_factory=aml.AML_EMBEDDING_FACTORY
        original_write=aml.service.write_event
        def factory(user):
            obj=original_factory(user);real=obj.embed
            def embed(texts):
                if any("first blocked" in text for text in texts):
                    embedding.set()
                    assert release.wait(5)
                return real(texts)
            obj.embed=embed
            return obj
        def write(*args,**kwargs):
            event=original_write(*args,**kwargs)
            if event.body=="second source":second_written.set()
            return event
        with patch.object(aml,"AML_EMBEDDING_FACTORY",side_effect=factory),patch.object(aml.service,"write_event",side_effect=write):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(self.add,"r1","A","s",["first blocked"])
                self.assertTrue(embedding.wait(5))
                second=pool.submit(self.add,"r2","A","s",["second source"])
                self.assertTrue(second_written.wait(5))
                release.set()
                self.assertEqual([first.result(10)[0],second.result(10)[0]],[200,200])
