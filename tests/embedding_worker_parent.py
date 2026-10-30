"""Synthetic parent-death fixture. Never a production entrypoint."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from tiger_memory.embedding_supervisor import NetworkSupervisor,P3bConfig
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.test_embedding_budget import policy

root,endpoint,stage=sys.argv[1:];root=Path(root)
worker=Path(sys.modules[NetworkSupervisor.__module__].__file__).with_name('embedding_worker.py')
budget=PersistentEmbeddingBudget.initialize(root/'ledger',policy(endpoint=endpoint))
s=NetworkSupervisor(P3bConfig(enabled=True,expected_worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),g_net=0,g_term=.05,g_reap=.1))

def pause(h=None):
    if h is None:h=next(iter(s._handles.values()))
    print(json.dumps({'pid':os.getpid(),'worker_pid':h.process.pid,'ledger_id':budget.ledger_id,'stage':stage}),flush=True)
    signal.pause()

original_write=s._write

def write(h,value,limit,until,admission=True):
    if stage=='before-init' and value['type']=='INIT':pause(h)
    if stage=='partial-permit' and value['type']=='PERMIT':
        from tiger_memory.embedding_supervisor import frame
        os.write(h.process.stdin.fileno(),frame(value,limit)[:57]);pause(h)
    return original_write(h,value,limit,until,admission)
s._write=write
original_reserve=budget.reserve

def reserve(*args):
    if stage=='ready':pause()
    result=original_reserve(*args)
    if stage=='reserved':pause()
    return result
budget.reserve=reserve
if stage=='sending':
    s._observer=lambda e:pause() if e['event']=='worker-sending' else None
p=DashScopeEmbeddingProvider(endpoint,'SYNTHETIC_PARENT86',root/'cache',64,
    allow_test_loopback=True,budget=budget,checkpoint_namespace='parent86',network_supervisor=s,max_elapsed_seconds=.6)
p.embed(['synthetic parent death'])
raise AssertionError('expected pause')
