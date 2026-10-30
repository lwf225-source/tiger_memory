"""Opt-in single-parent P3b supervisor. No credentials or bodies in events.

Owns every signal/wait/slot transition. No persistence, automatic resend or
cross-parent resource guarantee. Only macOS is enabled in this first version.
"""
from collections import deque
from dataclasses import dataclass, field
import base64
import hashlib
import math
import json
import struct
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import uuid

from .providers import ProviderUnavailable
# Intentionally duplicated framing: never import/execute worker before hash validation.
MAGIC = b'TM86'
HEADER = struct.Struct('!4sI32s')
INIT_LIMIT = 16384
PERMIT_LIMIT = 1024 * 1024
RESULT_LIMIT = 6 * 1024 * 1024
RESPONSE_LIMIT = 4 * 1024 * 1024
BOOT_IDLE_SECONDS = 5.0  # Synthetic opt-in startup bound, not a production SLO.


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('protocol')
        result[key] = value
    return result


def decode_json(data):
    return json.loads(data, object_pairs_hook=unique_object,
                      parse_constant=lambda _v: (_ for _ in ()).throw(ValueError('protocol')))


def frame(value, limit):
    body = json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
    if len(body) > limit:
        raise ValueError('protocol')
    return HEADER.pack(MAGIC, len(body), hashlib.sha256(body).digest()) + body


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)



def fail(code):
    raise ProviderUnavailable('embedding-worker-' + code)


@dataclass(frozen=True)
class P3bConfig:
    enabled: bool = False
    slots: int = 1
    quarantine_cap: int = 1
    ready_timeout: float = 2.0
    g_net: float = 0.0
    g_term: float = .1
    g_reap: float = .2
    g_shutdown: float = .5
    expected_worker_sha256: str = ''

    def validate(self):
        if type(self.enabled) is not bool:
            fail('configuration')
        if not self.enabled:
            return self
        if sys.platform != 'darwin':
            fail('platform-unvalidated')
        if (type(self.slots) is not int or not 1 <= self.slots <= 64
                or type(self.quarantine_cap) is not int or not 1 <= self.quarantine_cap <= self.slots
                or not sha(self.expected_worker_sha256)):
            fail('configuration')
        for key in ('ready_timeout','g_net','g_term','g_reap','g_shutdown'):
            value = getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 5:
                fail('configuration')
        if self.ready_timeout <= 0:
            fail('configuration')
        return self


@dataclass
class RequestContext:
    generation: int
    fingerprint: str
    batch_digest: str
    provider_deadline: float
    budgets: tuple = ()
    cancel_observed: float | None = None

    def validate(self):
        if (type(self.generation) is not int or not 0 <= self.generation <= 2**63-1
                or not sha(self.fingerprint) or not sha(self.batch_digest)
                or type(self.provider_deadline) not in (int,float) or not math.isfinite(self.provider_deadline)):
            fail('context')
        return self

    @property
    def admission_deadline(self):
        return min([self.provider_deadline] + [b.deadline for b in self.budgets])

    def check(self):
        for budget in self.budgets:
            budget.check('embedding-worker-admission')
        if time.monotonic() >= self.admission_deadline:
            fail('deadline')

    def cutoff(self, grace):
        now = time.monotonic()
        for budget in self.budgets:
            if now >= budget.deadline:
                budget.cancel('deadline')
            if budget.cancelled.is_set() and self.cancel_observed is None:
                self.cancel_observed = now
        result = min(self.provider_deadline, self.admission_deadline + grace)
        if self.cancel_observed is not None:
            result = min(result, self.cancel_observed + grace)
        return result


@dataclass
class Handle:
    context: RequestContext
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)
    lock: object = field(default_factory=threading.RLock)
    process: object = None
    life: object = None
    phase: str = 'allocated'
    attempt: str | None = None
    permit_used: bool = False
    accepting: bool = True
    accepted: bool = False
    quarantined: bool = False
    reaped: bool = False

    def binding(self):
        return dict(version=1,generation=self.context.generation,nonce=self.nonce,
                    attempt=self.attempt,fingerprint=self.context.fingerprint,batch_digest=self.context.batch_digest)


class NetworkSupervisor:
    def __init__(self, config: P3bConfig, observer=None):
        if not isinstance(config,P3bConfig):fail('configuration')
        self.config = config.validate()
        self._observer = observer
        self._lock = threading.RLock()
        self._handles = {}
        self._events = deque(maxlen=1024)
        self._closed = False
        self._owner_pid = os.getpid()
        self._observer_errors = 0
        self._code = None
        if config.enabled:
            path = Path(__file__).with_name('embedding_worker.py')
            try:
                if path.is_symlink():fail('code-integrity')
                with path.open('rb') as stream:code = stream.read(64*1024+1)
                if len(code)>64*1024 or hashlib.sha256(code).hexdigest()!=config.expected_worker_sha256:
                    fail('code-integrity')
                self._code=code.decode('utf-8')
            except (OSError,UnicodeError):fail('code-integrity')

    def _event(self, handle, event, **details):
        entry={'event':event,'monotonic':time.monotonic(),**handle.binding(),**details}
        with self._lock:self._events.append(entry)
        if self._observer is not None:
            try:self._observer(dict(entry))
            except Exception:
                with self._lock:self._observer_errors+=1

    def snapshot(self):
        with self._lock:
            return {'enabled':self.config.enabled,'closed':self._closed,'parent_pid':self._owner_pid,
                    'active_slots':len(self._handles),'quarantined':sum(h.quarantined for h in self._handles.values()),
                    'slot_limit':self.config.slots,'quarantine_cap':self.config.quarantine_cap,
                    'observer_errors':self._observer_errors,'events':[dict(e) for e in self._events],
                    'scope':'one-parent-lifetime','worker_sha256':self.config.expected_worker_sha256}

    def _spawn(self, life_read):
        # Only the read end is inherited. Popen's stdio pipes are private IPC;
        # all caller lock/DB/listener descriptors are closed during exec.
        return subprocess.Popen([sys.executable,'-I','-B','-c',self._code,str(life_read),
                                 self.config.expected_worker_sha256],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
            cwd='/',env={'PATH':os.defpath},close_fds=True,pass_fds=(life_read,))

    def _active(self,h):
        if not h.accepting:fail('abandoned')

    def _write(self,h,value,limit,until,admission=True):
        try:data=frame(value,limit)
        except (ValueError,TypeError,OverflowError):fail('frame')
        offset=0
        while offset<len(data):
            if admission:h.context.check()
            if time.monotonic()>=until:fail('deadline')
            with h.lock:
                self._active(h)
                if h.process is None or h.process.stdin is None:fail('pipe')
                fd=h.process.stdin.fileno()
            try:ready=select.select([],[fd],[],min(.02,max(0.,until-time.monotonic())))[1]
            except (OSError,ValueError):fail('pipe')
            if not ready:continue
            with h.lock:
                self._active(h)
                if admission:h.context.check()
                if time.monotonic()>=until:fail('deadline')
                try:offset+=os.write(h.process.stdin.fileno(),data[offset:offset+65536])
                except BlockingIOError:continue
                except (OSError,ValueError):fail('pipe')

    def _read(self,h,limit,until):
        data=bytearray();length=HEADER.size;checksum=None
        while len(data)<length:
            cutoff=until()
            if time.monotonic()>=cutoff:fail('deadline')
            with h.lock:
                self._active(h);fd=h.process.stdout.fileno()
            try:ready=select.select([fd],[],[],min(.02,max(0.,cutoff-time.monotonic())))[0]
            except (OSError,ValueError):fail('pipe')
            if not ready:continue
            with h.lock:
                self._active(h)
                cutoff=until()
                if time.monotonic()>=cutoff:fail('deadline')
                try:part=os.read(h.process.stdout.fileno(),min(65536,length-len(data)))
                except BlockingIOError:continue
                except (OSError,ValueError):fail('pipe')
            if not part:fail('truncated')
            data.extend(part)
            if checksum is None and len(data)==HEADER.size:
                magic,size,checksum=HEADER.unpack(data)
                if magic!=MAGIC or not 0<size<=limit:fail('frame')
                length+=size
        body=bytes(data[HEADER.size:])
        if hashlib.sha256(body).digest()!=checksum:fail('frame')
        try:value=decode_json(body)
        except (ValueError,UnicodeError,TypeError,RecursionError):fail('frame')
        cutoff=until()
        if not isinstance(value,dict) or time.monotonic()>=cutoff:fail('frame')
        return value

    @staticmethod
    def _matches(h,obj):
        return all(type(obj.get(k)) is type(v) and obj.get(k)==v for k,v in h.binding().items())

    def prepare(self,context,*,allow_loopback=False):
        if not self.config.enabled:fail('disabled')
        if os.getpid()!=self._owner_pid:fail('inherited-supervisor')
        context.validate();context.check();self.reap_pending()
        h=Handle(context)
        with self._lock:
            if (self._closed or len(self._handles)>=self.config.slots
                    or sum(v.quarantined for v in self._handles.values())>=self.config.quarantine_cap):
                fail('capacity')
            self._handles[h.nonce]=h
        deadline=min(context.admission_deadline,time.monotonic()+self.config.ready_timeout)
        try:
            with h.lock:
                self._active(h);context.check()
                read,write=os.pipe();h.life=write
                os.set_inheritable(write,False)
                try:
                    h.process=self._spawn(read)
                    os.set_blocking(h.process.stdin.fileno(),False)
                    os.set_blocking(h.process.stdout.fileno(),False)
                finally:os.close(read)
            h.phase='started';self._event(h,'started',pid=h.process.pid,code_sha256=self.config.expected_worker_sha256)
            self._write(h,{**h.binding(),'type':'INIT','admit_until':context.admission_deadline,
                           'accept_until':context.cutoff(self.config.g_net),'allow_loopback':allow_loopback,
                           'code_sha256':self.config.expected_worker_sha256},INIT_LIMIT,deadline)
            def ready_until():context.check();return deadline
            result=self._read(h,INIT_LIMIT,ready_until)
            expected=set(h.binding())|{'type','pid','platform','code_sha256','monotonic','open_fds','lifeline_fd'}
            if (set(result)!=expected or result['type']!='READY' or not self._matches(h,result)
                    or type(result['pid']) is not int or result['pid']!=h.process.pid or result['platform']!=sys.platform
                    or result['code_sha256']!=self.config.expected_worker_sha256
                    or type(result['monotonic']) not in (int,float)
                    or not time.monotonic()-5 <= result['monotonic'] <= time.monotonic()+.1
                    or type(result['lifeline_fd']) is not int or result['lifeline_fd']<=2
                    or not isinstance(result['open_fds'],list) or len(result['open_fds'])!=4
                    or any(type(fd) is not int for fd in result['open_fds'])
                    or set(result['open_fds'])!={0,1,2,result['lifeline_fd']}):
                fail('ready')
            context.check();h.phase='ready'
            self._event(h,'ready',pid=result['pid'],worker_monotonic=result['monotonic'],
                        open_fds=result['open_fds'],lifeline_fd=result['lifeline_fd'])
            return h
        except BaseException:
            self.cleanup(h)
            raise

    def validate_request(self,endpoint,credential,body):
        # Preflight real encoded size BEFORE spawning/reserving. Do not log it.
        if (not isinstance(body,bytes) or len(body)>PERMIT_LIMIT or not isinstance(endpoint,str) or len(endpoint.encode())>2048
                or not isinstance(credential,str) or not 1<=len(credential)<=4096
                or any(ord(c)<33 or ord(c)>126 for c in credential)):
            fail('request-size')
        probe=dict(version=1,type='PERMIT',generation=2**63-1,nonce='f'*32,attempt='f'*32,
                   fingerprint='f'*64,batch_digest='f'*64,endpoint=endpoint,credential=credential,
                   body_b64=base64.b64encode(body).decode('ascii'),socket_timeout=60.0)
        try:frame(probe,PERMIT_LIMIT)
        except (ValueError,TypeError):fail('request-size')

    def send(self,h,attempt,endpoint,credential,body,socket_timeout):
        try:
            self.validate_request(endpoint,credential,body)
            with h.lock:
                self._active(h)
                if (h.phase!='ready' or h.permit_used or not isinstance(attempt,str) or len(attempt)!=32
                        or any(c not in '0123456789abcdef' for c in attempt)
                        or hashlib.sha256(body).hexdigest()!=h.context.batch_digest):fail('permit')
                h.context.check();h.attempt=attempt;h.permit_used=True;h.phase='permit'
            self._event(h,'permit-start')
            self._write(h,{**h.binding(),'type':'PERMIT','endpoint':endpoint,'credential':credential,
                           'body_b64':base64.b64encode(body).decode('ascii'),'socket_timeout':socket_timeout},
                        PERMIT_LIMIT,h.context.admission_deadline)
            self._event(h,'permit-complete')
            sending=False
            while True:
                result=self._read(h,RESULT_LIMIT,lambda:h.context.cutoff(self.config.g_net))
                if not self._matches(h,result):fail('binding')
                if result.get('type')=='SENDING':
                    if (sending or set(result)!=set(h.binding())|{'type','monotonic'}
                            or type(result['monotonic']) not in (int,float)
                            or not math.isfinite(result['monotonic'])):fail('order')
                    sending=True;self._event(h,'worker-sending',worker_monotonic=result['monotonic']);continue
                if not sending or result.get('type')!='RESULT':fail('order')
                if result.get('status')=='ok':
                    if set(result)!=set(h.binding())|{'type','status','body_b64'}:fail('frame')
                    try:raw=base64.b64decode(result['body_b64'],validate=True)
                    except (ValueError,TypeError):fail('frame')
                    if len(raw)>RESPONSE_LIMIT:fail('frame')
                elif result.get('status')=='error':
                    if set(result)!=set(h.binding())|{'type','status','code'} or not isinstance(result['code'],str) or result['code'] not in {'network','http','response-too-large'}:fail('frame')
                    raw=None
                else:fail('frame')
                with h.lock:
                    self._active(h)
                    cutoff=h.context.cutoff(self.config.g_net)
                    if time.monotonic()>=cutoff:fail('deadline')
                    h.accepted=True;h.accepting=False;h.phase='accepted'
                self._event(h,'result-accepted',status=result['status'])
                if raw is None:fail(result['code'])
                return raw
        finally:self.cleanup(h)

    def _poll(self,h):
        return h.process.poll()

    def _wait(self,h,timeout):
        try:return h.process.wait(timeout=max(0.,timeout))
        except subprocess.TimeoutExpired:return None

    def _signal(self,h,kill=False):
        # Sole signal/reap owner under h.lock; never signal a numeric stale PID.
        if self._poll(h) is not None:return
        try:
            if kill:h.process.kill()
            else:h.process.terminate()
            self._event(h,'kill-requested' if kill else 'terminate-requested')
        except (OSError,ProcessLookupError):self._event(h,'signal-unconfirmed')

    def cleanup(self,h,deadline=None):
        # Cleanup faults must never replace an accepted response or free an
        # unconfirmed process slot. Only a confirmed wait/poll can release it.
        try:
            self._cleanup(h,deadline)
        except Exception:
            with h.lock:
                h.accepting=False
                h.quarantined=True
                self._event(h,'cleanup-unconfirmed')

    def _cleanup(self,h,deadline=None):
        with h.lock:
            if h.reaped:return
            h.accepting=False
            if not h.accepted:h.phase='abandoned';self._event(h,'abandoned')
            for stream in ((h.process.stdin,h.process.stdout) if h.process is not None else ()):
                try:stream.close()
                except Exception:self._event(h,'stream-close-unconfirmed')
            if h.life is not None:
                try:os.close(h.life)
                except OSError:pass
                h.life=None
            if h.process is None:
                h.reaped=True
            else:
                def allowance(cap):return cap if deadline is None else min(cap,max(0.,deadline-time.monotonic()))
                code=self._poll(h)
                if code is None:
                    self._signal(h);code=self._wait(h,allowance(self.config.g_term))
                if code is None:
                    self._signal(h,kill=True);code=self._wait(h,allowance(self.config.g_reap))
                if code is not None:
                    h.reaped=True;h.quarantined=False;self._event(h,'reaped',exit_code=code)
                else:
                    h.quarantined=True;self._event(h,'quarantined')
            if h.reaped:
                with self._lock:self._handles.pop(h.nonce,None)

    def reap_pending(self):
        with self._lock:pending=[h for h in self._handles.values() if h.quarantined]
        for h in pending:
            with h.lock:
                if h.reaped:continue
                try:code=self._poll(h)
                except Exception:
                    self._event(h,'reap-unconfirmed');continue
                if code is not None:
                    h.reaped=True;h.quarantined=False;self._event(h,'reaped',exit_code=code)
                    with self._lock:self._handles.pop(h.nonce,None)

    def close(self):
        with self._lock:self._closed=True;handles=list(self._handles.values())
        deadline=time.monotonic()+self.config.g_shutdown
        for h in handles:self.cleanup(h,deadline)
        return self.snapshot()
