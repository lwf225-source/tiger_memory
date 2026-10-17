"""Offline reference preparation, never a production server or paid smoke runner.

Default: verify the separately shipped public manifest without reading secrets.
--rehearse: disposable synthetic P3a HTTP exercise; literal loopback only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import uuid

SCHEMA = "aml-local-reference-88-v1"
PROFILE = {"identity": "deployment-candidate-unadopted", "transport": "P3a",
           "embedding": "text-embedding-v4", "dimensions": 1024,
           "retrieval_profile": "lexical-with-vectors", "session_rrf": True,
           "options_as_query_terms": True, "adjacent": 2, "session_views": False,
           "query_rewrite": "off", "recency": False, "reranker": "off",
           "fact_projection": "off", "deadline_enabled": False,
           "production_enabled": False, "old75_adoption_gate": "FAIL"}
MISSING = ["candidate_adoption", "host_os_python_resources", "store_and_tls",
           "approved_budget_receipt", "supplier_credential", "service_credential",
           "public_commit", "deployment_authorization", "official_access_and_run_authorization"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def support_file(name):
    package = Path(__file__).resolve().parent
    for root in (package / "_bundle", package.parent, package.parent.parent):
        path = root / "scripts" / name
        if path.is_file():
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("reference-binding-invalid")
            return path
    raise ValueError("reference-binding-missing")


def self_check():
    """No imports of service/config, environment secret reads, writes or probes."""
    try:
        manifest = json.loads(support_file("aml_reference_manifest.json").read_bytes())
        package = Path(__file__).resolve().parent
        expected = manifest["runtime"]
        if manifest["schema"] != SCHEMA or manifest["profile"] != PROFILE:
            raise ValueError()
        if manifest["profile_sha256"] != digest(PROFILE) or not isinstance(expected, dict):
            raise ValueError()
        actual_names = {p.name for p in package.glob("*.py")}
        if actual_names != set(expected):
            raise ValueError()
        for name, checksum in expected.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+\.py", name):
                raise ValueError()
            path = package / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError()
        for name, checksum in manifest["support"].items():
            if not re.fullmatch(r"aml_[A-Za-z0-9_.]+", name):
                raise ValueError()
            if hashlib.sha256(support_file(name).read_bytes()).hexdigest() != checksum:
                raise ValueError()
        return {"passed": True, "mode": "static", "schema": SCHEMA,
                "profile": PROFILE, "profile_sha256": digest(PROFILE),
                "runtime_files_verified": len(expected), "manifest_sha256": hashlib.sha256(
                    support_file("aml_reference_manifest.json").read_bytes()).hexdigest(),
                "missing_production_facts": MISSING, "production_ready": False,
                "secrets_read": False, "network_probe": False,
                "identity_equivalent_to75": "UNPROVEN"}
    except (OSError, ValueError, KeyError, TypeError):
        return {"passed": False, "mode": "static", "reason": "reference-binding-invalid",
                "production_ready": False, "secrets_read": False, "network_probe": False}


class NetworkFence:
    """Process-wide audit hook: install only in this dedicated CLI process.

    No DNS names, non-INET sockets, foreign bind/connect, exec or shell helpers.
    Native sandbox used by acceptance adds an independent OS boundary.
    """
    def __init__(self, loopback=False):
        self.loopback = loopback
        self.allowed = 0
        self.denied = 0

    def __call__(self, event, args):
        if event in {"os.system", "os.exec", "os.posix_spawn"}:
            self.denied += 1
            raise PermissionError("reference-subprocess-denied")
        # The service requires local Git probes. The CLI cleans Git environment;
        # acceptance OS sandbox also constrains the child process.
        if event == "subprocess.Popen":
            argv = args[1]
            if not isinstance(argv, (list, tuple)) or not argv or str(argv[0]).split('/')[-1] != 'git':
                self.denied += 1
                raise PermissionError("reference-subprocess-denied")
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            host = args[0] if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"} else args[1][0] if isinstance(args[1], tuple) else None
            if not self.loopback or host != "127.0.0.1":
                self.denied += 1
                raise PermissionError("reference-network-denied")
            self.allowed += 1

    def positive_control(self):
        blocked = 0
        for host in ("198.51.100.1", "example.invalid"):
            try:
                socket.getaddrinfo(host, 443)
            except PermissionError:
                blocked += 1
            else:
                raise RuntimeError("reference-guard-control-failed")
        return blocked == 2


class PrivateLog:
    """Bounded acknowledged local evidence, no unbounded request-side disk wait.

    On queue/disk/ack/rotation failure latch unhealthy and refuse new work.
    A stuck daemon may survive until CLI exit; no hard FS deadline is claimed.
    Two 64KiB files, 0600, private new directory; last two rotations retained.
    """
    def __init__(self, directory, config_sha256, *, artifact_sha256, wait=.25, capacity=16, max_bytes=65536):
        if (not isinstance(config_sha256, str) or not re.fullmatch("[0-9a-f]{64}", config_sha256)
                or not isinstance(artifact_sha256, str) or not re.fullmatch("[0-9a-f]{64}", artifact_sha256)
                or type(wait) not in (int,float) or not math.isfinite(wait) or not 0 < wait <= 1
                or type(capacity) is not int or not 1 <= capacity <= 64
                or type(max_bytes) is not int or not 1024 <= max_bytes <= 65536):
            raise ValueError("reference-log-configuration")
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        self.path = self.directory / "events.jsonl"
        self.config = config_sha256
        self.artifact = artifact_sha256
        self.boot = uuid.uuid4().hex
        self.wait, self.max_bytes = wait, max_bytes
        self.failed = threading.Event()
        self.closed = False
        self.written = 0
        self.q = queue.Queue(capacity)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _append(self, raw):
        if self.path.exists() and self.path.stat().st_size + len(raw) > self.max_bytes:
            os.replace(self.path, self.directory / "events.previous.jsonl")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise OSError("reference-log-permissions")
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0: raise OSError("reference-log-write")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _run(self):
        while True:
            item = self.q.get()
            if item is None: return
            raw, done = item
            try:
                if not self.failed.is_set():
                    self._append(raw)
                    self.written += 1
            except Exception:
                self.failed.set()
            finally:
                done.set()

    def record(self, phase, *, audit_id=None, operation="lifecycle", status=0,
               elapsed_ms=0, business_completed=False, write_completed=False,
               attempts=(), recovery="none"):
        if (phase not in {"boot", "begin", "end", "stop"} or operation not in {"add", "search", "health", "status", "other", "lifecycle"}
                or recovery not in {"none", "cache_or_no_work", "settled", "unknown", "blocked_or_failed"}
                or type(status) is not int or not 0 <= status <= 599
                or type(elapsed_ms) is not int or elapsed_ms < 0
                or type(business_completed) is not bool or type(write_completed) is not bool
                or audit_id is not None and not re.fullmatch('[0-9a-f]{32}', audit_id)):
            raise ValueError("reference-log-schema")
        attempts = list(attempts)
        if len(attempts) > 100 or any(not isinstance(a, str) or not re.fullmatch('[0-9a-f]{32}', a) for a in attempts):
            raise ValueError("reference-log-schema")
        if self.closed or self.failed.is_set(): return False
        entry = {"schema": SCHEMA, "at": datetime.now(timezone.utc).isoformat(),
                 "boot_id": self.boot, "config_sha256": self.config, "artifact_sha256": self.artifact, "phase": phase,
                 "audit_id": audit_id, "operation": operation, "status": status,
                 "elapsed_ms": elapsed_ms, "business_completed": business_completed,
                 "server_write_completed": write_completed, "attempts": attempts, "recovery": recovery}
        raw = (json.dumps(entry, sort_keys=True) + '\n').encode()
        if len(raw) > min(self.max_bytes, 16384):
            self.failed.set(); return False
        done = threading.Event()
        try: self.q.put_nowait((raw, done))
        except queue.Full: self.failed.set(); return False
        if not done.wait(self.wait): self.failed.set(); return False
        return not self.failed.is_set()

    def close(self):
        self.closed = True
        try: self.q.put_nowait(None)
        except queue.Full: self.failed.set()
        self.thread.join(self.wait)
        if self.thread.is_alive(): self.failed.set()
        return {"healthy": not self.failed.is_set(), "writer_stopped": not self.thread.is_alive(),
                "written": self.written, "boot_id": self.boot}


def reference_handler(aml, local, log):
    class Handler(aml.AMLRequestHandler):
        def _send_json(self, status, payload):
            self._observed_status = status
            return super()._send_json(status, payload)

        def _observed(self, call):
            audit = uuid.uuid4().hex
            started = time.monotonic()
            self._observed_status = 0
            self._business_completed = self._response_delivered = False
            local.providers = []
            operation = self.path.split('?', 1)[0].strip('/')
            if operation not in {'add','search','health','status'}: operation = 'other'
            if not log.record('begin', audit_id=audit, operation=operation):
                self.close_connection = True
                self._send_json(503, {'detail': {'reason': 'reference-evidence-unavailable'}})
                return
            try:
                call()
            finally:
                providers = local.providers
                attempts = [a for p in providers for a in p.attempts]
                recovery = ('unknown' if any(a['state']=='UNKNOWN' for a in attempts) else
                            'settled' if attempts else 'blocked_or_failed' if self._observed_status >= 400 else 'cache_or_no_work')
                log.record('end', audit_id=audit, operation=operation, status=self._observed_status,
                           elapsed_ms=int((time.monotonic()-started)*1000),
                           business_completed=self._business_completed,
                           write_completed=self._response_delivered,
                           attempts=[a['id'] for a in attempts], recovery=recovery)
                local.providers = []

        def do_GET(self): self._observed(super().do_GET)
        def do_POST(self): self._observed(super().do_POST)
    return Handler


def make_reference_server(aml, local, log):
    server = aml.make_server('127.0.0.1', 0, max_connections=2)
    # Controlled finite clients only. Join handlers before stop/log close so an
    # already-written HTTP response cannot race its pending end evidence.
    # This is NOT a hard shutdown deadline or a production server change.
    server.daemon_threads = False
    server.RequestHandlerClass = reference_handler(aml, local, log)
    return server


def rehearse(fence, artifact_sha256):
    """Disposable local reference. Service-object restart, not process-kill proof."""
    from unittest.mock import patch

    key, user, text = 'SYNTHETIC88_KEY_CANARY', 'SYNTHETIC88_USER_CANARY', 'SYNTHETIC88_BODY_CANARY violet boxes'
    received = []
    mode = {'fail': False}
    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(len(payload['input']))
            if mode['fail']:
                self.send_response(503); raw=b'SYNTHETIC88_UPSTREAM_ERROR_CANARY'
            else:
                self.send_response(200)
                raw=json.dumps({'model':'text-embedding-v4','usage':{'total_tokens':len(payload['input'])},
                    'data':[{'index':i,'embedding':[1.0, float(len(t)%11+1)] + [0.0]*1022}
                            for i,t in enumerate(payload['input'])]}).encode()
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    with tempfile.TemporaryDirectory(prefix='tiger-reference88-') as folder:
        root=Path(folder).resolve(); home=root/'home';home.mkdir(mode=0o700)
        with patch.dict(os.environ, {'HOME':str(home),'PATH':os.defpath,'TIGER_MEMORY_DIR':str(root/'store'),
                'AML_API_KEY':key,'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}, clear=True):
            from . import aml_adapter as aml
            from .embedding_budget import BudgetPolicy, PersistentEmbeddingBudget
            from .remote_embeddings import DashScopeEmbeddingProvider
            from .providers import ProviderUnavailable
            remote=ThreadingHTTPServer(('127.0.0.1',0),Fake)
            rt=threading.Thread(target=remote.serve_forever,kwargs={'poll_interval':.02},daemon=True);rt.start()
            endpoint='http://127.0.0.1:%d/embeddings'%remote.server_port
            policy=BudgetPolicy(endpoint=endpoint,region='test-only',model='text-embedding-v4',dimensions=1024,
                token_limit_per_row=8192,price_nano_cny_per_token=1,max_cost_nano_cny=100000000,
                max_requests=100,evidence_sha256=hashlib.sha256(b'synthetic-only-no-real-approval').hexdigest(),test_only=True)
            budget=PersistentEmbeddingBudget.initialize(root/'ledger',policy)
            receipt=root/'fake-receipt.json';receipt.write_text(json.dumps(budget.receipt()));receipt.chmod(0o600)
            with ExitStack() as stack:
                local=threading.local()
                def factory(uid):
                    p=DashScopeEmbeddingProvider(endpoint,key,cache_dir=aml.user_storage_dir(uid)/'embedding-cache',
                        dimensions=1024,allow_test_loopback=True,budget=budget,
                        checkpoint_namespace=aml.user_storage_dir(uid).name)
                    local.providers.append(p)
                    return p
                stack.enter_context(patch.multiple(aml,AML_EMBEDDING_FACTORY=factory,AML_EMBEDDING='remote-v4',
                    AML_STRICT_RETRIEVAL=True,AML_PROFILE='lexical',AML_QUERY_REWRITE='off',AML_FACT_PROJECTION='off',
                    AML_RERANKER='off',AML_RECENCY_BOOST=False,AML_SESSION_RRF=True,
                    AML_RUNTIME_CONFIG={'profile':PROFILE,'config_sha256':digest(PROFILE)}))
                public=[]; boot=[]; rows=[]; server=None; thread=None; log=None
                def start(index):
                    nonlocal server,thread,log
                    log=PrivateLog(root/('logs%d'%index),digest(PROFILE),artifact_sha256=artifact_sha256)
                    # Reopened validated synthetic journal, with no live writer yet.
                    with closing(sqlite3.connect((budget.root/'ledger.sqlite3').as_uri()+'?mode=ro',uri=True)) as conn:
                        pending=[r[0] for r in conn.execute("SELECT attempt FROM attempts WHERE state='UNKNOWN' ORDER BY seq")]
                    if not log.record('boot',attempts=pending,recovery='unknown' if pending else 'none'):
                        raise RuntimeError('reference-log-boot')
                    server=make_reference_server(aml,local,log)
                    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.02},daemon=True);thread.start()
                def stop():
                    nonlocal server
                    if server is not None:
                        server.shutdown();server.server_close();thread.join(3);server=None
                        log.record('stop');boot.append(log.close())
                        for path in sorted(log.directory.glob('*.jsonl')):
                            rows.extend(json.loads(line) for line in path.read_text().splitlines())
                def request(path,payload=None,auth=True):
                    c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                    try:
                        c.request('GET' if payload is None else 'POST',path,
                            body=json.dumps(payload) if payload is not None else None,
                            headers={'Authorization':'Bearer '+key} if auth else {})
                        r=c.getresponse();body=json.loads(r.read());status=r.status
                        if status>=400 or path in {'/health','/status'}:public.append(body)
                        return status,body
                    finally:c.close()
                def add(rid,body):return request('/add',{'request_id':rid,'user_id':user,'session_id':'synthetic-session',
                    'messages':[{'role':'user','content':body}]})
                try:
                    start(1)
                    assert request('/health')[0]==200
                    assert request('/status',auth=False)[0]==401
                    assert request('/status')[0]==200
                    assert add('ok',text)[0]==200
                    query={'query':'violet','user_id':user,'top_k':100}
                    first=request('/search',query);assert first[0]==200 and first[1]['data']
                    baseline=budget.snapshot();count=len(received)
                    stop()
                    budget=PersistentEmbeddingBudget.from_receipt(receipt,allow_test_policy=True)
                    start(2)
                    assert add('ok','changed first-write')[0]==200
                    assert request('/search',query)==first
                    assert len(received)==count and budget.snapshot()==baseline
                    mode['fail']=True
                    assert add('unknown','unavailable distinct batch')[0]==503
                    assert budget.snapshot()['unknown']==1
                    mode['fail']=False
                    with patch.object(budget,'settle',side_effect=ProviderUnavailable('synthetic-settle-failed')):
                        assert add('raw-unknown','valid checkpoint retained')[0]==503
                    blocked=budget.snapshot();unknown_count=len(received)
                    assert blocked['unknown']==2
                    stop()
                    budget=PersistentEmbeddingBudget.from_receipt(receipt,allow_test_policy=True)
                    start(3);mode['fail']=False
                    assert add('unknown','unavailable distinct batch')[0]==503
                    assert add('changed-id','unavailable distinct batch')[0]==503
                    assert len(received)==unknown_count and budget.snapshot()==blocked
                    assert add('raw-unknown','valid checkpoint retained')[0]==200
                    assert add('ok','ignored changed input')[0]==200
                    assert len(received)==unknown_count and budget.snapshot()==blocked
                    rejected=False
                    try:PersistentEmbeddingBudget.from_receipt(receipt)
                    except ProviderUnavailable:rejected=True
                    assert rejected
                finally:
                    stop();remote.shutdown();remote.server_close();rt.join(3)
                exposed=json.dumps({'events':rows,'public':public})
                assert not any(c in exposed for c in (key,user,text,'SYNTHETIC88_UPSTREAM_ERROR_CANARY'))
                assert all(b['healthy'] and b['writer_stopped'] for b in boot)
                assert len({b['boot_id'] for b in boot})==3
                return {'passed':True,'mode':'synthetic-loopback','profile':PROFILE,
                    'config_sha256':digest(PROFILE),'service_restarts':2,'process_restart_tested':False,
                    'fake_requests':len(received),'unknown_retained':blocked['unknown'],
                    'retry_new_requests':len(received)-unknown_count,'budget_unchanged_on_retry':budget.snapshot()==blocked,
                    'raw_cache_reused':True,'valid_raw_unknown_reused':True,'missing_raw_unknown_blocked':True,
                    'test_receipt_production_rejected':rejected,
                    'log':boot,'events':rows,'sensitive_public_canary_hits':0,
                    'production_ready':False,'quality_gate75':'FAIL','official_smoke':False,
                    'guard':{'positive_control':True,'denied':fence.denied,'loopback_operations':fence.allowed}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--self-check',action='store_true')
    action.add_argument('--rehearse',action='store_true')
    args=parser.parse_args(argv)
    fence=NetworkFence(loopback=args.rehearse);sys.addaudithook(fence)
    checked=self_check()
    if not checked['passed']:
        print(json.dumps(checked,sort_keys=True));return 2
    try:
        if not fence.positive_control():raise RuntimeError()
        result=rehearse(fence, checked["manifest_sha256"]) if args.rehearse else checked
        if not args.rehearse:result['network_guard_positive_control']=True
    except Exception:
        result={'passed':False,'mode':'synthetic-loopback' if args.rehearse else 'static',
                'reason':'reference-check-failed','production_ready':False}
    print(json.dumps(result,sort_keys=True))
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
