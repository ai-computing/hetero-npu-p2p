"""CPU regressions for DMA ownership, route identity and aliased destinations."""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest import mock

from dmabuf_fakes import FakeGpu, FakeNative
from hetero_p2p import gpu as gpu_module
from hetero_p2p.gpu import CudaRegion, DmabufCopy, RngdDmabufBackend, plan_copies
from hetero_p2p.pd_dmabuf import DmabufPDTransport
from test_dmabuf_adapters import Lease

try:
    from rm_core import KVLayout, MemoryTier, TransportError
    from hetero_p2p.airm_dmabuf import DmabufBinding, DmabufKVTransport
except ImportError:
    DmabufKVTransport = None


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.gpu = FakeGpu(32768)
        self.native = FakeNative(self.gpu, dram=32768)
        self.backend = RngdDmabufBackend(None, native=self.native)
        self.region = self.gpu.region(exact_cache=1)

    def tearDown(self):
        # The fake's explicit completion stands in for a CQE, never a hardware fence.
        self.native.quiesce = lambda ctx: True
        if hasattr(self.backend, 'recover'):
            self.backend.recover()
        self.region.close()
        self.backend.close()

    def test_read_rejects_overlap_across_distinct_exact_exports_before_export(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.backend.copy('read', (DmabufCopy(0, self.region, 0, 8192),
                                       DmabufCopy(16384, self.region, 4096, 4096)))
        self.assertEqual(self.region.exports_made, 0)
        self.assertEqual(len(self.native.runs), 0)

    def test_read_rejects_overlap_across_region_views_of_same_cuda_address(self):
        view = CudaRegion(self.gpu.base + 4096, 8192, self.gpu, exporter=self.gpu.exporter)
        try:
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                plan_copies('read', (DmabufCopy(0, self.region, 0, 8192), DmabufCopy(16384, view, 0, 4096)))
            self.assertEqual((self.region.exports_made, view.exports_made), (0, 0))
        finally:
            view.close()

    def test_reads_allow_adjacent_destinations_and_same_address_on_distinct_devices(self):
        plan_copies('read', (DmabufCopy(0, self.region, 0, 4096), DmabufCopy(8192, self.region, 4096, 4096))).validate()
        other = self.gpu.region(device=1)
        try:
            plan_copies('read', (DmabufCopy(0, self.region, 0, 4096), DmabufCopy(8192, other, 0, 4096))).validate()
        finally:
            other.close()

    def test_uncertain_completion_retains_owner_pins_and_exports_until_cqe(self):
        class Owner:
            pass
        owner = Owner()
        reference = weakref.ref(owner)
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.backend.copy('read', (DmabufCopy(0, self.region, 0, 4096),), owner=owner, cache=False)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertFalse(self.backend.recover())
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.region.close()
        self.region.release_pins()
        self.assertTrue(self.region._pinned)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.backend.close()
        self.assertFalse(self.native.closed)
        self.native.quiesce = lambda ctx: True
        self.assertTrue(self.backend.recover())
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(self.region._pinned)

    def test_pd_recover_refuses_unknown_completion_and_retains_request(self):
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with mock.patch('hetero_p2p.pd_dmabuf.resolve'):
            transport = DmabufPDTransport(self.backend, lambda s, d: (DmabufCopy(0, self.region, 0, 4096),),
                                          source='cuda:0', destination='rngd:0', spec='s')
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            transport.transfer('request', Lease('s', object(), None, 'cuda:0'), Lease('s', object(), None, 'rngd:0'))
        self.assertFalse(transport.recover())
        with self.assertRaises(RuntimeError):
            transport.finish('request')
        with self.assertRaisesRegex(RuntimeError, 'already transferred'):
            transport.transfer('request', Lease('s', 1, None, 'cuda:0'), Lease('s', 1, None, 'rngd:0'))
        self.native.quiesce = lambda ctx: True
        self.assertTrue(transport.recover())

    def test_cached_plan_retains_the_current_request_owner_on_uncertain_completion(self):
        class Owner:
            pass
        self.backend.read(0, self.region, length=4096, owner=Owner())
        owner = Owner()
        reference = weakref.ref(owner)
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with self.assertRaises(RuntimeError):
            self.backend.read(0, self.region, length=4096, owner=owner)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())

    def test_dropping_backend_cannot_release_an_uncertain_owner(self):
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with self.assertRaises(RuntimeError):
            self.backend.read(0, self.region, length=4096, cache=False)
        reference = weakref.ref(self.backend)
        self.backend = None
        gc.collect()
        self.backend = reference()  # restore the fixture so teardown can inject its completion
        self.assertIsNotNone(self.backend)

    def test_failed_release_region_remains_registered_until_recovery(self):
        key = (0, self.region.ptr, self.region.size)
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with self.assertRaises(RuntimeError):
            self.backend.read(0, self.region, length=4096)
        with mock.patch.dict(gpu_module._REGIONS, {key: self.region}, clear=True), \
                mock.patch.object(gpu_module, '_tensor_range', return_value=key):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                gpu_module.release_region(object())
            self.native.quiesce = lambda ctx: True
            self.backend.recover()
            gpu_module.release_region(object())
            self.assertTrue(self.region.closed)

    def test_another_backend_cannot_reuse_a_cached_quarantined_region(self):
        other = RngdDmabufBackend(None, native=FakeNative(self.gpu, dram=32768))
        self.addCleanup(other.close)
        other.read(0, self.region, length=4096)
        self.native.run = lambda ctx, args: (-4096, 0)
        self.native.quiesce = lambda ctx: False
        with self.assertRaises(RuntimeError):
            self.backend.read(0, self.region, length=4096)
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            other.read(0, self.region, length=4096)

    def test_interruption_without_native_receipt_quarantines_owners(self):
        def interrupted(ctx, args):
            raise KeyboardInterrupt()
        self.native.run = interrupted
        self.native.quiesce = lambda ctx: False
        try:
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                self.backend.read(0, self.region, length=4096)
        except KeyboardInterrupt:
            self.fail('interruption escaped without quarantining DMA owners')
        self.assertFalse(self.backend.recover())
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            self.region.close()


@unittest.skipIf(DmabufKVTransport is None, 'rm-core not installed')
class RouteIdentityTests(unittest.TestCase):
    def test_plan_rejects_same_vendor_wrong_index_in_both_directions(self):
        gpu = FakeGpu(32768)
        region = gpu.region()
        layout = KVLayout(num_layers=1, num_kv_heads=8, head_dim=128, dtype='bf16')
        try:
            for source_rngd in (False, True):
                for npu_index, gpu_index in ((3, 0), (0, 2)):
                    with self.subTest(source_rngd=source_rngd, npu=npu_index, gpu=gpu_index):
                        transport = DmabufKVTransport(device='/dev/rngd/npu0ch0')
                        npu = MemoryTier('rngd', 'device-hbm', 48 << 30, f'host-a/furiosa:{npu_index}')
                        cuda = MemoryTier('cuda', 'device-hbm', 80 << 30, f'host-a/nvidia:{gpu_index}')
                        src, dst = (npu, cuda) if source_rngd else (cuda, npu)
                        transport.bind('r', DmabufBinding(0, region))
                        with mock.patch('hetero_p2p.airm_dmabuf.resolve'):
                            with self.assertRaisesRegex(TransportError, 'device|identity'):
                                transport.plan('r', src, dst, layout, layout, (0, 1))
                        self.assertEqual(region.exports_made, 0)
        finally:
            region.close()

    def test_nonzero_matching_devices_are_accepted(self):
        gpu = FakeGpu(32768)
        region = gpu.region(device=2)
        layout = KVLayout(num_layers=1, num_kv_heads=8, head_dim=128, dtype='bf16')
        npu = MemoryTier('rngd', 'device-hbm', 48 << 30, 'host-a/furiosa:3')
        cuda = MemoryTier('cuda', 'device-hbm', 80 << 30, 'host-a/nvidia:2')
        try:
            for src, dst in ((cuda, npu), (npu, cuda)):
                transport = DmabufKVTransport(device='/dev/rngd/npu3ch0')
                transport.bind('r', DmabufBinding(0, region))
                with mock.patch('hetero_p2p.airm_dmabuf.resolve'):
                    plan = transport.plan('r', src, dst, layout, layout, (0, 1))
                self.assertEqual(plan.nbytes, 4096)
        finally:
            region.close()


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('gcc'), 'needs Linux and GCC; no devices used')
class NativeFaultTests(unittest.TestCase):
    def test_fatal_submission_and_failed_drain_report_uncertain_completion(self):
        fixtures = Path(__file__).parent / 'native_dmabuf'
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'faults'
            build = subprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-I', str(fixtures),
                                    str(fixtures / 'faults.c'), '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
