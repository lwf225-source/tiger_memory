from pathlib import Path
import sys,io,json,tempfile,unittest,time,threading,http.server
from unittest.mock import patch
from urllib.error import HTTPError
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P/'base')]
import runtime_loader
aml,remote,config=runtime_loader.install()
from tiger_memory.providers import ProviderUnavailable
class Response(io.BytesIO):status=200
def response():return Response(json.dumps({'model':'text-embedding-v4','data':[{'index':0,'embedding':[1.0]*64}]}).encode())
class Opener:
 def __init__(self,items):self.items=list(items);self.calls=0
 def open(self,r,timeout):
  assert timeout is None
  self.calls+=1;x=self.items.pop(0)
  if isinstance(x,Exception):raise x
  return x
class Cases(unittest.TestCase):
 def provider(self,cache=None):return remote.DashScopeEmbeddingProvider('https://www.dmxapi.cn/v1/embeddings','SYNTHETIC',cache,dimensions=64)
 def test_timeout_recovers_and_cache_replay_no_send(self):
  with tempfile.TemporaryDirectory() as t:
   p=self.provider(Path(t).resolve());fake=Opener([TimeoutError(),response()]);p._opener=fake
   with patch.object(remote,'sleep') as sleep:
    result=p.embed(['synthetic']);self.assertEqual(sleep.call_args.args,(1.0,))
   self.assertEqual(p.embed(['synthetic']),result);self.assertEqual(fake.calls,2)
 def test_503_and_429_backoff(self):
  p=self.provider();p._opener=Opener([HTTPError('https://example.invalid',503,'unavailable',{},None),HTTPError('https://example.invalid',429,'busy',{'Retry-After':'3'},None),response()])
  with patch.object(remote,'sleep') as sleep:
   self.assertEqual(len(p.embed(['synthetic'])),1);self.assertEqual([x.args[0] for x in sleep.call_args_list],[1.0,3.0])
 def test_auth_error_not_retried(self):
  p=self.provider();p._opener=Opener([HTTPError('https://example.invalid',401,'no',{},None)])
  with self.assertRaises(ProviderUnavailable):p.embed(['synthetic'])
  self.assertEqual(p._opener.calls,1)
 def test_invalid_response_not_retried(self):
  p=self.provider();p._opener=Opener([Response(b'{}')])
  with self.assertRaises(ProviderUnavailable):p.embed(['synthetic'])
  self.assertEqual(p._opener.calls,1)
 def test_real_response_after_31_seconds(self):
  class Handler(http.server.BaseHTTPRequestHandler):
   def do_POST(self):
    self.rfile.read(int(self.headers['Content-Length']));time.sleep(31.2)
    body=response().getvalue();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
   def log_message(self,*args):pass
  server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   p=remote.DashScopeEmbeddingProvider('http://127.0.0.1:'+str(server.server_port)+'/embeddings','SYNTHETIC',dimensions=64,allow_test_loopback=True)
   begin=time.monotonic();self.assertEqual(len(p.embed(['synthetic delayed response'])),1);elapsed=time.monotonic()-begin
   self.assertGreater(elapsed,30);self.assertEqual(p.request_batches,1)
   print(json.dumps({'delayed_response_seconds':elapsed,'sent_batches':p.request_batches,'timeout':p.timeout}),flush=True)
  finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main(verbosity=2)
