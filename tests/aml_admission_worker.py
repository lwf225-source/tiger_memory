"""Two independent synthetic AML HTTP process fixture; no external transport."""
import json
import os
from pathlib import Path
import socket
import sys
import time
from tiger_memory import aml_adapter as aml, service
from tiger_memory.embedding_budget import PersistentEmbeddingBudget
from tiger_memory.remote_embeddings import DashScopeEmbeddingProvider
from tests.test_embedding_budget import FakeTransport


def main(root, role):
    root=Path(root)
    def audit(event,args):
        if event in ('socket.connect','socket.bind') and (not isinstance(args[1],tuple) or args[1][0] not in ('127.0.0.1','::1')):
            raise RuntimeError('external denied')
        if event in ('socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr') and args[0] not in ('127.0.0.1','::1','localhost'):
            raise RuntimeError('DNS denied')
    sys.addaudithook(audit)
    socket.getfqdn=lambda host='':'localhost'
    budget=PersistentEmbeddingBudget.from_receipt(root/'receipt.json', allow_test_policy=True)
    for name,value in {'AML_EMBEDDING':'remote-v4','AML_STRICT_RETRIEVAL':True,'AML_RERANKER':'off','AML_QUERY_REWRITE':'off','AML_FACT_PROJECTION':'off','AML_RECENCY_BOOST':False,'AML_NARRATIVE_TIME':False,'AML_SESSION_RRF':False,'AML_SESSION_VIEWS':False}.items():setattr(aml,name,value)
    def factory(user):
        obj=DashScopeEmbeddingProvider(budget.policy.endpoint,'SYNTHETIC_ONLY',dimensions=64,
            allow_test_loopback=True,budget=budget,cache_dir=aml.user_storage_dir(user)/'embedding-cache',
            checkpoint_namespace=user,max_request_batches=1)
        obj._opener=FakeTransport(wire_file=root/'wire.log')
        original=obj.plan
        def plan(texts):
            result=original(texts)
            (root/('plan-'+role)).write_text(json.dumps({'chunks':result['total_chunks'],
                'lock_inode':(aml.user_storage_dir(user)/'.aml-add.lock').stat().st_ino}))
            return result
        obj.plan=plan
        return obj
    aml.AML_EMBEDDING_FACTORY=factory
    original=service.write_event
    def write(*args,**kwargs):
        event=original(*args,**kwargs)
        if role=='A' and event.body.startswith('a'*100):
            (root/'first-committed').write_text('ready')
            end=time.monotonic()+8
            while not (root/'release').exists():
                if time.monotonic()>=end:raise RuntimeError('fixture release timeout')
                time.sleep(.01)
        return event
    service.write_event=write
    server=aml.make_server('127.0.0.1',0)
    (root/('port-'+role)).write_text(str(server.server_port))
    server.serve_forever(poll_interval=.02)

if __name__=='__main__':main(*sys.argv[1:])
