"""The two GPU <-> RNGD adapters over a fake native library: Optimus Prime PD Transport (structural) and the
rm-core KVTransport `p2p-dmabuf`."""
import os, random, tempfile, unittest
from dataclasses import dataclass
from typing import Any
from unittest import mock

from dmabuf_fakes import FakeGpu, FakeNative, MiB
from hetero_p2p import paths
from hetero_p2p.gpu import DmabufCopy, RngdDmabufBackend
from hetero_p2p.pd_dmabuf import DmabufPDTransport

try:
    from rm_core import KVLayout, MemoryTier, NoHardware, TransportError
    from rm_core.conformance import check_kv_transport
    from hetero_p2p.airm_dmabuf import DmabufBinding, DmabufKVTransport
except ImportError:  # rm-core is a declared dependency; tolerate checkouts that lack it
    DmabufKVTransport = None

OPEN = 'NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04  Release Build\n'
PROPRIETARY = 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Release Build\n'


@dataclass(frozen=True)
class Lease:                       # the fields of opt_prime.accel.pd.contracts.KVLease the transport reads
    spec: Any
    owner: Any
    payload: Any
    endpoint: str


class ModuleMixin:
    def module(self, text):
        f = tempfile.NamedTemporaryFile('w', delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        patcher = mock.patch.object(paths, 'NVIDIA_VERSION_FILE', f.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fakes(self):
        self.g = FakeGpu(32 * MiB)
        self.native = FakeNative(self.g)
        self.backend = RngdDmabufBackend(None, native=self.native)
        self.region = self.g.region(slice_bytes=16 * MiB)
        self.addCleanup(self.region.close)


class PDTransportTests(ModuleMixin, unittest.TestCase):
    SPEC = ('model', 'rev', 'bf16', 'token-major-v1')

    def setUp(self):
        self.module(OPEN)
        self.fakes()
        self.notified = []
        # payloads: the GPU KV is region bytes [0, 8 MiB); the RNGD KV is DRAM [16 MiB, +8 MiB)
        self.builder = lambda s, d: [DmabufCopy(16 * MiB, self.region, 0, 8 * MiB)]

    def transport(self, source='cuda:0', destination='rngd:/dev/rngd/npu0ch0'):
        return DmabufPDTransport(self.backend, self.builder, self.notified.append, source=source,
                                 destination=destination, spec=self.SPEC)

    def test_gpu_prefill_to_rngd_decode_and_back(self):
        data = random.Random(1).randbytes(8 * MiB)
        self.g.mem[:8 * MiB] = data
        t = self.transport()
        src, dst = Lease(self.SPEC, object(), None, 'cuda:0'), Lease(self.SPEC, object(), None, 'rngd:/dev/rngd/npu0ch0')
        self.assertTrue(t.supports('cuda:0', 'rngd:/dev/rngd/npu0ch0', self.SPEC))
        receipt = t.transfer('r1', src, dst)
        self.assertEqual((receipt['direction'], receipt['transport'], receipt['bytes_requested']), ('write', 'p2p-dmabuf', 8 * MiB))
        self.assertEqual(self.notified, [dst])
        self.assertEqual(bytes(self.native.dram[16 * MiB:24 * MiB]), data)
        with self.assertRaisesRegex(RuntimeError, 'already transferred'):
            t.transfer('r1', src, dst)
        t.finish('r1')
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            t.finish('r1')
        back = self.transport('rngd:/dev/rngd/npu0ch0', 'cuda:0')
        self.g.mem[:8 * MiB] = bytes(8 * MiB)
        self.assertEqual(back.transfer('r2', dst, src)['direction'], 'read')
        self.assertEqual(bytes(self.g.mem[:8 * MiB]), data)

    def test_failure_does_not_notify_and_can_recover(self):
        self.native.fail_after = 0
        t = self.transport()
        src, dst = Lease(self.SPEC, object(), None, 'cuda:0'), Lease(self.SPEC, object(), None, 'rngd:/dev/rngd/npu0ch0')
        with self.assertRaises(RuntimeError):
            t.transfer('r1', src, dst)
        self.assertEqual(self.notified, [])
        self.assertTrue(t.recover())

    def test_route_and_format_checks(self):
        t = self.transport()
        other = ('model', 'rev', 'dlfloat16', 'token-major-v1')
        with self.assertRaisesRegex(ValueError, 'physical format'):
            t.transfer('r', Lease(other, 1, None, 'cuda:0'), Lease(self.SPEC, 1, None, 'rngd:/dev/rngd/npu0ch0'))
        for s, d in (('cuda:0', 'cuda:1'), ('rngd:0', 'rngd:1'), ('atom:0', 'rngd:0')):
            with self.assertRaises(ValueError):
                self.transport(s, d)

    def test_proprietary_module_is_rejected_at_construction(self):
        self.module(PROPRIETARY)
        with self.assertRaises(paths.ProprietaryModuleError):
            self.transport()


@unittest.skipIf(DmabufKVTransport is None, 'rm-core not installed')
class KVTransportTests(ModuleMixin, unittest.TestCase):
    def setUp(self):
        self.module(OPEN)
        self.fakes()
        self.gpu_tier = MemoryTier('a100-0', 'device-hbm', 80 << 30, 'host-a/nvidia:0')
        self.npu_tier = MemoryTier('rngd0', 'device-hbm', 48 << 30, 'host-a/furiosa:0')
        # 1 layer x 8 KV heads x 128 dims bf16 = 4 KiB per token
        self.layout = KVLayout(num_layers=1, num_kv_heads=8, head_dim=128, dtype='bf16')
        self.metrics = []
        self.t = DmabufKVTransport(backend=self.backend, metric_sink=self.metrics.append)

    def plan(self, src, dst, tokens=(0, 16), rid='r'):
        self.t.bind(rid, DmabufBinding(4 * MiB, self.region, 0))
        return self.t.plan(rid, src, dst, self.layout, self.layout, tokens)

    def test_supports_only_same_host_gpu_rngd(self):
        self.assertTrue(self.t.supports(self.gpu_tier, self.npu_tier))
        self.assertTrue(self.t.supports(self.npu_tier, self.gpu_tier))
        self.assertFalse(self.t.supports(self.gpu_tier, MemoryTier('x', 'device-hbm', 1, 'host-b/furiosa:0')))
        self.assertFalse(self.t.supports(self.npu_tier, MemoryTier('atom', 'device-dram', 1, 'host-a/rebellions:0')))
        self.assertFalse(self.t.supports(self.gpu_tier, MemoryTier('host', 'host-dram', 1)))

    def test_both_directions_byte_exact_with_metrics(self):
        data = random.Random(2).randbytes(64 * 1024)
        self.g.mem[16 * 4096:32 * 4096] = data[:64 * 1024]
        h = self.t.submit(self.plan(self.gpu_tier, self.npu_tier, (16, 32)))
        r = self.t.wait(h, 1.0)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.nbytes, 16 * 4096)
        self.assertEqual(bytes(self.native.dram[4 * MiB + 16 * 4096:4 * MiB + 32 * 4096]), data)
        self.g.mem[16 * 4096:32 * 4096] = bytes(64 * 1024)
        self.assertTrue(self.t.wait(self.t.submit(self.plan(self.npu_tier, self.gpu_tier, (16, 32))), 1.0).ok)
        self.assertEqual(bytes(self.g.mem[16 * 4096:32 * 4096]), data)
        self.assertEqual([m.name for m in self.metrics], ['kv_transfer_ms', 'kv_transfer_gbps'] * 2)
        self.assertTrue(all(m.tags['kv_transport'] == 'p2p-dmabuf' for m in self.metrics))
        with self.assertRaises(TransportError):
            self.t.wait(h, 1.0)

    def test_plan_rejections(self):
        with self.assertRaisesRegex(TransportError, 'bind'):
            self.t.plan('none', self.gpu_tier, self.npu_tier, self.layout, self.layout, (0, 16))
        for tokens in ((16, 16), (-1, 4)):
            with self.assertRaises(TransportError):
                self.plan(self.gpu_tier, self.npu_tier, tokens)
        other = KVLayout(num_layers=1, num_kv_heads=8, head_dim=128, dtype='fp16')
        self.t.bind('x', DmabufBinding(0, self.region))
        with self.assertRaisesRegex(TransportError, 'identical'):
            self.t.plan('x', self.gpu_tier, self.npu_tier, self.layout, other, (0, 16))
        small = KVLayout(num_layers=1, num_kv_heads=1, head_dim=8, dtype='bf16')       # 32 bytes per token
        self.t.bind('y', DmabufBinding(0, self.region))
        with self.assertRaisesRegex(TransportError, '4 KiB'):
            self.t.plan('y', self.gpu_tier, self.npu_tier, small, small, (0, 16))
        with self.assertRaisesRegex(TransportError, 'exceed'):
            self.plan(self.gpu_tier, self.npu_tier, (0, 32 * MiB // 4096 + 1))

    def test_proprietary_module_fails_plan_explicitly(self):
        self.module(PROPRIETARY)
        with self.assertRaisesRegex(TransportError, 'open kernel module'):
            self.plan(self.gpu_tier, self.npu_tier)

    def test_backend_failure_is_not_retried(self):
        self.native.fail_after = 0
        r = self.t.wait(self.t.submit(self.plan(self.gpu_tier, self.npu_tier)), 1.0)
        self.assertFalse(r.ok)
        self.assertIn('DMA-BUF PDMA failed', r.error)
        self.assertEqual(len(self.native.runs), 1)
        self.assertEqual(self.metrics, [])

    def test_rm_core_conformance_with_fake_backend(self):
        self.t.bind('conf-1', DmabufBinding(0, self.region))
        layout = KVLayout(num_layers=2, num_kv_heads=8, head_dim=128, dtype='bf16')     # 8 KiB per token
        check_kv_transport(self.t, tiers=(self.gpu_tier, self.npu_tier), layout=layout)

    def test_conformance_case_skips_without_hardware(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(NoHardware):
                DmabufKVTransport(device='/nonexistent/rngd').conformance_case()
            with self.assertRaisesRegex(NoHardware, 'HETERO_P2P_DMABUF_LIB'):
                DmabufKVTransport(device=__file__).conformance_case()
            with self.assertRaisesRegex(NoHardware, 'FIXTURE'):
                DmabufKVTransport(library=__file__, device=__file__).conformance_case()
        self.module(PROPRIETARY)
        with self.assertRaisesRegex(NoHardware, 'open kernel module'):
            DmabufKVTransport(library=__file__, device=__file__).conformance_case()


if __name__ == '__main__':
    unittest.main()
