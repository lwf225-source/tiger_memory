"""Local admission file validation, deadline semantics and descriptor lifecycle."""
import math
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from unittest.mock import patch
from tiger_memory import aml_add_lock as lock
from tests.test_embedding_budget import evidence

class AdmissionFileTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.path=self.root/'.aml-add.lock'

    def test_private_stable_inode_noninherit_and_exception_closes_fd(self):
        descriptors=[];original=os.open
        def opened(*args,**kwargs):
            fd=original(*args,**kwargs);descriptors.append(fd);return fd
        with patch.object(lock.os,'open',side_effect=opened):
            with lock.add_admission_lock(self.root) as observation:
                self.assertTrue(observation['acquired'])
                self.assertFalse(os.get_inheritable(descriptors[-1]))
                self.assertEqual(stat.S_IMODE(self.path.stat().st_mode),0o600)
                inode=self.path.stat().st_ino
            with self.assertRaises(OSError):os.fstat(descriptors[-1])
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with lock.add_admission_lock(self.root):raise RuntimeError('synthetic')
            with self.assertRaises(OSError):os.fstat(descriptors[-1])
            with lock.add_admission_lock(self.root):self.assertEqual(self.path.stat().st_ino,inode)
        self.assertTrue(self.path.exists())

    def test_symlink_fifo_directory_and_wrong_mode_owner_are_rejected(self):
        target=self.root/'target';target.write_text('preserve')
        self.path.symlink_to(target)
        with self.assertRaises(lock.AddLockUnavailable):
            with lock.add_admission_lock(self.root):pass
        self.assertEqual(target.read_text(),'preserve');self.path.unlink()
        os.mkfifo(self.path,0o600)
        start=time.monotonic()
        with self.assertRaises(lock.AddLockUnavailable):
            with lock.add_admission_lock(self.root):pass
        self.assertLess(time.monotonic()-start,1);self.path.unlink()
        self.path.mkdir()
        with self.assertRaises(lock.AddLockUnavailable):
            with lock.add_admission_lock(self.root):pass
        self.path.rmdir();self.path.write_text('');self.path.chmod(0o644)
        with self.assertRaises(lock.AddLockUnavailable):
            with lock.add_admission_lock(self.root):pass
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode),0o644)
        self.path.chmod(0o600)
        with patch.object(lock.os,'getuid',return_value=os.getuid()+1),self.assertRaises(lock.AddLockUnavailable):
            with lock.add_admission_lock(self.root):pass

    def test_timeout_validation_zero_and_late_acquisition(self):
        for value in (-1,True,float('nan'),float('inf'),10**400,'5'):
            with self.subTest(value=repr(value)),self.assertRaises(lock.AddLockUnavailable):
                with lock.add_admission_lock(self.root,value):pass
        with lock.add_admission_lock(self.root,0):
            with self.assertRaises(lock.AddBusy):
                with lock.add_admission_lock(self.root,0):pass
        entered=False
        with patch.object(lock,'monotonic',side_effect=[10.0,15.01]),self.assertRaises(lock.AddBusy):
            with lock.add_admission_lock(self.root,5):entered=True
        self.assertFalse(entered)
        with lock.add_admission_lock(self.root,0):pass

    def test_real_five_second_wait_is_local_busy_not_business_admission(self):
        with lock.add_admission_lock(self.root):
            start=time.monotonic()
            with self.assertRaises(lock.AddBusy) as caught:
                with lock.add_admission_lock(self.root,5):self.fail('unexpected admission')
            elapsed=time.monotonic()-start
        self.assertGreaterEqual(elapsed,5)
        self.assertLess(elapsed,6.5)
        self.assertGreaterEqual(caught.exception.elapsed_seconds,5)
        evidence("admission-five-second-wait", {"configured_seconds": 5, "elapsed_seconds": elapsed,
            "exception_wait_seconds": caught.exception.elapsed_seconds, "tolerance_seconds": 1.5,
            "acquired": False, "expired": True, "whole_request_deadline": False})
        with lock.add_admission_lock(self.root,0):pass
