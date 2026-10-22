import os, tempfile, unittest
from unittest import mock

from hetero_p2p import paths
from hetero_p2p.paths import PathUnavailable, ProprietaryModuleError, lookup, resolve

OPEN = 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04  Release Build\n'
PROPRIETARY = 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Release Build\n'


class PathTableTests(unittest.TestCase):
    def module(self, text):
        f = tempfile.NamedTemporaryFile('w', delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return mock.patch.object(paths, 'NVIDIA_VERSION_FILE', f.name)

    def test_gpu_rngd_is_rngd_initiated_both_ways(self):
        with self.module(OPEN):
            for src, dst, mech in (('cuda', 'rngd', 'DMABUF_WRITE'), ('rngd', 'cuda', 'DMABUF_READ')):
                p = resolve(src, dst)
                self.assertEqual((p.initiator, p.backend, p.requires), ('rngd', 'RngdDmabufBackend', ('nvidia-open-module',)))
                self.assertIn(mech, p.mechanism)

    def test_proprietary_module_fails_explicitly(self):
        with self.module(PROPRIETARY):
            for pair in (('cuda', 'rngd'), ('rngd', 'cuda')):
                with self.assertRaisesRegex(ProprietaryModuleError, 'open kernel module.*No host-staged fallback'):
                    resolve(*pair)
            self.assertEqual(resolve('cuda', 'rngd', check=False).status, 'supported')
            self.assertEqual(resolve('rngd', 'atom').backend, 'RngdPdmaBackend')   # unaffected by the GPU driver
        self.assertEqual(paths.nvidia_kernel_module('/nonexistent'), None)

    def test_blocked_and_unvalidated_pairs_never_fall_back(self):
        for src, dst in (('cuda', 'atom'), ('atom', 'cuda'), ('rngd', 'rngd'), ('atom', 'rngd'), ('cuda', 'host'),
                         ('host', 'rngd'), ('cuda', 'cuda'), ('atom', 'atom')):
            with self.assertRaises(PathUnavailable):
                resolve(src, dst)
        self.assertEqual(lookup('rngd', 'host').status, 'blocked')
        self.assertIn('staging', lookup('rngd', 'host').reason)
        with self.assertRaises(KeyError):
            lookup('tpu', 'rngd')


if __name__ == '__main__':
    unittest.main()
