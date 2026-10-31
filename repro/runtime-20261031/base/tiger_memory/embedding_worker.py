"""Standalone P3b one-exchange worker. Standard library only; no store access.

Launched from hash-verified code bytes using python -I -B -c. Private IPC is
not a public log. The watchdog is cooperative with OS scheduling, not realtime.
"""
import base64
import hashlib
import json
import math
import os
import select
import struct
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

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


def context_ok(obj, attempt):
    return (obj.get('version') == 1 and type(obj.get('version')) is int
            and type(obj.get('generation')) is int and 0 <= obj['generation'] <= 2**63-1
            and isinstance(obj.get('nonce'), str) and len(obj['nonce']) == 32
            and all(c in '0123456789abcdef' for c in obj['nonce'])
            and sha(obj.get('fingerprint')) and sha(obj.get('batch_digest'))
            and (obj.get('attempt') is None if attempt is None else obj.get('attempt') == attempt))


def public_context(obj):
    return {k: obj[k] for k in ('version', 'generation', 'nonce', 'attempt', 'fingerprint', 'batch_digest')}


def receive(fd, limit, deadline):
    data = bytearray()
    length = HEADER.size
    expected = None
    while len(data) < length:
        if time.monotonic() >= deadline[0]:
            raise ValueError('expired')
        if not select.select([fd], [], [], min(.02, max(0., deadline[0]-time.monotonic())))[0]:
            continue
        try:
            part = os.read(fd, min(65536, length-len(data)))
        except BlockingIOError:
            continue
        if not part:
            raise ValueError('protocol')
        data.extend(part)
        if expected is None and len(data) == HEADER.size:
            magic, size, expected = HEADER.unpack(data)
            if magic != MAGIC or size > limit or not size:
                raise ValueError('protocol')
            length += size
    body = bytes(data[HEADER.size:])
    if hashlib.sha256(body).digest() != expected:
        raise ValueError('protocol')
    value = decode_json(body)
    if not isinstance(value, dict):
        raise ValueError('protocol')
    return value


def write(fd, value, limit, deadline):
    data = frame(value, limit)
    offset = 0
    while offset < len(data):
        if time.monotonic() >= deadline[0]:
            raise ValueError('expired')
        if not select.select([], [fd], [], min(.02, max(0., deadline[0]-time.monotonic())))[1]:
            continue
        try:
            offset += os.write(fd, data[offset:offset+65536])
        except BlockingIOError:
            continue


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    life = int(sys.argv[1])
    expected_code = sys.argv[2]
    original = sys.orig_argv
    actual_code = original[original.index('-c')+1].encode()
    if hashlib.sha256(actual_code).hexdigest() != expected_code:
        return 72
    deadline = [time.monotonic() + BOOT_IDLE_SECONDS]
    os.set_blocking(life, False)
    os.set_blocking(0, False)
    os.set_blocking(1, False)

    def supervise():
        while True:
            if time.monotonic() >= deadline[0]:
                os._exit(73)
            try:
                readable = select.select([life], [], [], .02)[0]
                if readable and not os.read(life, 1):
                    os._exit(74)
                if readable:  # lifeline has no data protocol
                    os._exit(74)
            except (OSError, ValueError):
                os._exit(74)
    threading.Thread(target=supervise, daemon=True).start()
    init = receive(0, INIT_LIMIT, deadline)
    required = {'version','type','generation','nonce','attempt','fingerprint','batch_digest',
                'admit_until','accept_until','allow_loopback','code_sha256'}
    if (set(init) != required or init['type'] != 'INIT' or not context_ok(init, None)
            or init['code_sha256'] != expected_code or type(init['allow_loopback']) is not bool):
        return 72
    for key in ('admit_until', 'accept_until'):
        if type(init[key]) not in (int,float) or not math.isfinite(init[key]) or init[key] <= time.monotonic():
            return 72
    if init['accept_until'] < init['admit_until']:
        return 72
    deadline[0] = init['accept_until']
    context = public_context(init)
    # /dev/fd enumeration descriptor is closed before fstat checks.
    fds = []
    for name in os.listdir('/dev/fd'):
        try:
            fd = int(name); os.fstat(fd); fds.append(fd)
        except (OSError,ValueError):
            pass
    write(1, {**context, 'type':'READY', 'code_sha256':expected_code,
              'pid':os.getpid(), 'platform':sys.platform, 'monotonic':time.monotonic(),
              'open_fds':sorted(fds), 'lifeline_fd':life}, INIT_LIMIT, deadline)
    permit = receive(0, PERMIT_LIMIT, deadline)
    required = set(context) | {'type','endpoint','credential','body_b64','socket_timeout'}
    attempt = permit.get('attempt')
    if (set(permit) != required or permit['type'] != 'PERMIT'
            or not isinstance(attempt,str) or len(attempt) != 32
            or any(c not in '0123456789abcdef' for c in attempt)
            or not context_ok(permit, attempt)):
        return 72
    if any(permit[k] != v for k,v in context.items() if k != 'attempt'):
        return 72
    context['attempt'] = attempt
    endpoint, credential = permit['endpoint'], permit['credential']
    if (not isinstance(endpoint,str) or len(endpoint.encode()) > 2048
            or not isinstance(credential,str) or not 1 <= len(credential) <= 4096
            or any(ord(c) < 33 or ord(c) > 126 for c in credential)):
        return 72
    parsed = urlsplit(endpoint)
    if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(c.isspace() for c in endpoint)
            or not (parsed.scheme == 'https' or (init['allow_loopback'] and parsed.scheme == 'http'
                                                and parsed.hostname in {'127.0.0.1','::1'}))):
        return 72
    body = base64.b64decode(permit['body_b64'], validate=True)
    if hashlib.sha256(body).hexdigest() != context['batch_digest']:
        return 72
    payload = decode_json(body)
    if (set(payload) != {'model','input','dimensions','encoding_format'}
            or payload['model'] != 'text-embedding-v4' or payload['encoding_format'] != 'float'
            or type(payload['dimensions']) is not int or payload['dimensions'] not in {64,128,256,512,768,1024,1536,2048}
            or not isinstance(payload['input'],list) or not 1 <= len(payload['input']) <= 10
            or any(not isinstance(t,str) or not t or len(t.encode()) > 8192 for t in payload['input'])):
        return 72
    timeout = permit['socket_timeout']
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        return 72
    if time.monotonic() >= init['admit_until']:
        return 73
    # At most one permit is consumed, no loop and no hidden retry.
    write(1, {**context,'type':'SENDING','monotonic':time.monotonic()}, INIT_LIMIT, deadline)
    if time.monotonic() >= init['admit_until']:
        return 73
    status, code, raw = 'error', 'network', None
    try:
        request = Request(endpoint, data=body, method='POST', headers={
            'Authorization':'Bearer '+credential,'Content-Type':'application/json','Accept':'application/json'})
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout) as response:
            if response.status != 200:
                code = 'http'
            else:
                raw = response.read(RESPONSE_LIMIT+1)
                if len(raw) > RESPONSE_LIMIT:
                    code = 'response-too-large';raw = None
                else:
                    status = 'ok'
    except HTTPError as exc:
        exc.close();code = 'http'
    except Exception:
        code = 'network'
    result = {**context,'type':'RESULT','status':status}
    if status == 'ok':result['body_b64'] = base64.b64encode(raw).decode('ascii')
    else:result['code'] = code
    write(1,result,RESULT_LIMIT,deadline)
    return 0


if __name__ == '__main__':
    try:
        result = main()
    except BaseException:
        result = 72
    os._exit(result)
