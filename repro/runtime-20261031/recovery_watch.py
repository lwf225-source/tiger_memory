"""Linux local-filesystem notification guard; loss always requests a full scan.

Watches directories AND source inodes: writes through a hard-link alias must
invalidate metadata too. This is an accelerator, never a durable source index.
Unsupported platforms/filesystems and resource exhaustion use the full scanner.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import struct
import sys


class WatchUnavailable(Exception):
    pass


class WatchInvalidated(Exception):
    pass


class SourceWatch:
    # Allow the bounded 8192-source index plus watched ancestors/year/months.
    # Exhaustion still takes the authoritative full-scan path; no inode watch
    # is omitted (hard-link mutations must remain observable).
    MAX_WATCHES = 8704
    MAX_EVENT_BYTES = 4 * 1024 * 1024
    # Do not subscribe to ACCESS/OPEN/CLOSE_NOWRITE: reads are not mutations.
    MUTATIONS = 0x2 | 0x4 | 0x8 | 0x40 | 0x80 | 0x100 | 0x200
    SELF_LOSS = 0x400 | 0x800 | 0x2000 | 0x4000 | 0x8000
    ONLYDIR = 0x01000000
    DONT_FOLLOW = 0x02000000
    ISDIR = 0x40000000
    # Network/unknown filesystems may miss mutations made on another host.
    LOCAL_FS = {0xEF53, 0x58465342, 0x9123683E, 0x01021994, 0x794C7630}

    def __init__(self, root, tree_root=None):
        self.fd = -1
        self.tree_root = Path(tree_root or root)
        self.directory_identity = {}
        self.paths = {}
        self.watches = {}
        if sys.platform != 'linux':
            raise WatchUnavailable('platform')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]
        self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.libc.inotify_add_watch.restype = ctypes.c_int
        self.libc.fstatfs.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.root_device = Path(root).stat().st_dev
        self.mount_fingerprint = self._mount_fingerprint()
        descriptor = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            buffer = (ctypes.c_long * 64)()
            if self.libc.fstatfs(descriptor, buffer) != 0 or (buffer[0] & 0xFFFFFFFF) not in self.LOCAL_FS:
                raise WatchUnavailable('filesystem')
        finally:
            os.close(descriptor)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise WatchUnavailable('init')

    def add(self, path, directory=False):
        path = Path(path)
        if path in self.paths:
            return
        if len(self.paths) >= self.MAX_WATCHES:
            raise WatchUnavailable('watch_limit')
        if path.lstat().st_dev != self.root_device:
            raise WatchUnavailable("nested_filesystem")
        mask = self.MUTATIONS | self.SELF_LOSS | self.DONT_FOLLOW
        if directory:
            mask |= self.ONLYDIR
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), mask)
        if wd < 0:
            raise WatchUnavailable('watch_add')
        # Multiple names for one inode can share a wd. Preserve every alias.
        if directory:
            st = path.lstat()
            self.directory_identity[path] = (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid)
        self.paths[path] = wd
        self.watches.setdefault(wd, set()).add((path, directory))

    @staticmethod
    def _mount_fingerprint():
        return hashlib.sha256(Path("/proc/self/mountinfo").read_bytes()).digest()

    def changes(self):
        if self._mount_fingerprint() != self.mount_fingerprint:
            raise WatchInvalidated("mount_table_changed")
        changed = set()
        consumed = 0
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return changed
            except OSError as exc:
                raise WatchInvalidated('read') from exc
            if not data:
                raise WatchInvalidated('closed')
            consumed += len(data)
            if consumed > self.MAX_EVENT_BYTES:
                raise WatchInvalidated('event_limit')
            offset = 0
            while offset < len(data):
                if len(data) - offset < 16:
                    raise WatchInvalidated('truncated')
                wd, mask, cookie, length = struct.unpack_from('iIII', data, offset)
                offset += 16
                if offset + length > len(data):
                    raise WatchInvalidated('truncated_name')
                name = os.fsdecode(data[offset:offset + length].split(b'\0', 1)[0])
                offset += length
                if mask & self.SELF_LOSS or wd not in self.watches:
                    raise WatchInvalidated('lost_or_replaced')
                for parent, directory in self.watches[wd]:
                    if directory:
                        target = parent / name if name else parent
                        if mask & self.ISDIR or not name or target in self.directory_identity:
                            # Ignore sibling trees. Our ancestors and anything
                            # in the events tree remain structurally guarded.
                            relevant = (target in self.paths or
                                        target == self.tree_root or self.tree_root in target.parents)
                            if not relevant:
                                continue
                            if mask == 0x4 or mask == (0x4 | self.ISDIR):
                                # ensure_private_dir chmod(0700) emits ATTRIB
                                # even when identity/permissions are unchanged.
                                try:
                                    st = target.lstat()
                                    identity = (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid)
                                except OSError:
                                    raise WatchInvalidated('directory_missing')
                                if identity == self.directory_identity.get(target):
                                    continue
                            raise WatchInvalidated('directory_change')
                        # The shared root also contains SQLite WAL, manifests,
                        # and sibling projects. These cannot change this source
                        # tree and must not make sync/final guards look unstable.
                        # Keep explicitly watched inodes (including hard links)
                        # and every child in this events tree. Structural events
                        # above have already been checked before this filter.
                        if (target not in self.paths and target != self.tree_root
                                and self.tree_root not in target.parents):
                            continue
                        if target in self.paths and mask & (0x80 | 0x100):
                            # Old inode may survive via another hard link: never
                            # keep its watch bound to a replacement pathname.
                            raise WatchInvalidated('source_replaced')
                        changed.add(target)
                    else:
                        changed.add(parent)

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        self.directory_identity.clear()
        self.paths.clear()
        self.watches.clear()
