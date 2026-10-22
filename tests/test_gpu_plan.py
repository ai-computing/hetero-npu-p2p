"""CPU tests for hetero_p2p.gpu: region slicing and export caching, plan building, and the backend contract
(fence, receipt, no partial success) over a fake native library that really copies bytes."""
import os, random, tempfile, types, unittest
from unittest import mock

from dmabuf_fakes import FakeGpu, FakeNative, MiB
from hetero_p2p import gpu, paths
from hetero_p2p.gpu import (CudaRegion, DmabufCopy, DmabufPlan, DmabufSpan, PathUnavailable, RngdDmabufBackend,
                            plan_copies, plan_copy, SPAN)


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.g = FakeGpu(40 * MiB)
        self.r = self.g.region(slice_bytes=16 * MiB, exact_cache=2)

    def tearDown(self):
        self.r.close()

    def test_large_ranges_use_slices_split_at_boundaries(self):
        parts = self.r.locate(8 * MiB, 32 * MiB)            # larger than one slice -> slices 0, 1, 2
        self.assertEqual([(o, n) for _, o, n in parts], [(8 * MiB, 8 * MiB), (0, 16 * MiB), (0, 8 * MiB)])
        self.assertEqual(len({fd for fd, _, _ in parts}), 3)
        self.assertEqual([os.fstat(fd).st_size for fd, _, _ in parts], [16 * MiB, 16 * MiB, 8 * MiB])

    def test_small_ranges_export_exactly_once_and_are_reused(self):
        a = self.r.locate(15 * MiB, 2 * MiB)                 # crosses a slice boundary: still one exact export
        self.assertEqual(len(a), 1)
        self.assertEqual(os.fstat(a[0][0]).st_size, 2 * MiB)
        made = self.r.exports_made
        for _ in range(5):
            self.assertEqual(self.r.locate(15 * MiB, 2 * MiB), a)
        self.assertEqual(self.r.exports_made, made)          # cached: no new export

    def test_exact_cache_is_bounded_but_pinned_exports_survive(self):
        fds = [self.r.locate(k * MiB, MiB)[0][0] for k in range(4)]     # 4 > exact_cache=2, all pinned
        self.assertEqual(self.r.exports, 4)
        for fd in fds:
            os.fstat(fd)                                     # none closed while a plan may name them
        self.r.release_pins()
        self.r.locate(10 * MiB, MiB)                         # now the oldest unpinned ones are evicted
        self.assertEqual(self.r.exports, 2)

    def test_exact_cache_disabled_uses_slices(self):
        r = self.g.region(slice_bytes=16 * MiB, exact_cache=0)
        try:
            self.assertEqual(len(r.locate(15 * MiB, 2 * MiB)), 2)
        finally:
            r.close()

    def test_alignment_and_bounds(self):
        with self.assertRaises(ValueError):
            CudaRegion(4097, 4096, object())
        with self.assertRaises(ValueError):
            CudaRegion(4096, 4096, object(), slice_bytes=512 * MiB)
        with self.assertRaises(ValueError):
            CudaRegion(4096, 4096, None)
        for off, n in ((0, 0), (1, 4096), (0, 41 * MiB), (39 * MiB, 2 * MiB)):
            with self.assertRaises(ValueError):
                self.r.locate(off, n)

    def test_closed_region_refuses(self):
        self.r.close()
        with self.assertRaises(ValueError):
            self.r.locate(0, MiB)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.g = FakeGpu(40 * MiB)
        self.r = self.g.region(slice_bytes=16 * MiB, exact_cache=0)

    def tearDown(self):
        self.r.close()

    def test_spans_and_command_grouping(self):
        p = plan_copy('write', self.r, 0, 1 << 20, 40 * MiB, descriptors_per_command=4).validate()
        self.assertEqual(p.bytes, 40 * MiB)
        self.assertEqual(len(p.spans), 20)                   # 2 MiB spans
        self.assertEqual(p.commands, 2 + 2 + 1)              # 8, 8, 4 spans per 16/16/8 MiB slice, 4 per command
        self.assertEqual([g[2] for g in p.groups()], [4, 4, 4, 4, 4])
        self.assertEqual([s.npu_offset for s in p.spans[:2]], [1 << 20, (1 << 20) + SPAN])
        self.assertEqual(len({s.fd for s in p.spans}), 3)

    def test_plan_copies_concatenates(self):
        p = plan_copies('read', (DmabufCopy(0, self.r, 0, 4 * MiB), DmabufCopy(64 * MiB, self.r, 32 * MiB, 2 * MiB)),
                        descriptors_per_command=64)
        self.assertEqual(p.bytes, 6 * MiB)
        self.assertEqual(p.commands, 2)                      # different slices -> different FDs
        self.assertEqual(p.spans[-1].npu_offset, 64 * MiB)

    def test_alignment_bounds_and_overlap(self):
        fd = self.r.slice_fd(0)
        bad = [DmabufSpan(fd, 1, 0, 4096)], [DmabufSpan(fd, 0, 0, 3 * MiB)], [DmabufSpan(fd, 0, 16 * MiB, 4096)], \
              [DmabufSpan(fd, gpu.DRAM_LIMIT, 0, 4096)]
        for spans in bad:
            with self.assertRaises(ValueError):
                DmabufPlan('read', tuple(spans), object()).validate()
        with self.assertRaises(ValueError):
            DmabufPlan('sideways', (DmabufSpan(fd, 0, 0, 4096),), object()).validate()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            DmabufPlan('write', (DmabufSpan(fd, 0, 0, 8192), DmabufSpan(fd, 4096, 8192, 4096)), object()).validate()
        alias = os.dup(fd)
        try:
            with self.assertRaisesRegex(ValueError, 'overlapping'):   # FD aliases of one DMA-BUF
                DmabufPlan('read', (DmabufSpan(fd, 0, 0, 8192), DmabufSpan(alias, 8192, 4096, 4096)),
                           object()).validate()
        finally:
            os.close(alias)
        DmabufPlan('read', (DmabufSpan(fd, 0, 0, 8192), DmabufSpan(fd, 0, 8192, 4096)), object()).validate()


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.g = FakeGpu(80 * MiB)
        self.native = FakeNative(self.g)
        self.b = RngdDmabufBackend(None, '/dev/rngd/npu0ch0', queue_depth=4, descriptors_per_command=8,
                                   native=self.native)
        self.synced = []
        self.r = self.g.region(slice_bytes=16 * MiB, sync=lambda: self.synced.append(1))

    def tearDown(self):
        self.r.close()

    def test_round_trip_is_byte_exact_through_the_api(self):
        rng = random.Random(7)
        for n in (4096, 2 * MiB, 3 * MiB, 16 * MiB, 40 * MiB):
            data = rng.randbytes(n)
            self.g.mem[:n] = data
            w = self.b.write(self.r, 8 * MiB, length=n)       # GPU -> RNGD
            self.assertEqual(bytes(self.native.dram[8 * MiB:8 * MiB + n]), data)
            self.g.mem[40 * MiB:40 * MiB + n] = bytes(n)      # poison the destination
            r = self.b.read(8 * MiB, self.r, region_offset=40 * MiB, length=n)   # RNGD -> GPU
            self.assertEqual(bytes(self.g.mem[40 * MiB:40 * MiB + n]), data)
            for rec, d in ((w, 'write'), (r, 'read')):
                self.assertEqual((rec['direction'], rec['cqe_result'], rec['bytes_requested']), (d, 0, n))
                self.assertEqual(rec['completed'], rec['requested'])
        self.assertEqual(self.native.depth, 4)
        self.assertEqual(len(self.synced), 5)                 # sync before every GPU -> RNGD write, never for reads

    def test_cached_region_is_exported_once(self):
        for _ in range(3):
            self.b.write(self.r, 0, length=64 * MiB)
            self.b.read(0, self.r, length=2 * MiB)
        self.assertEqual(self.r.exports_made, 4 + 1)          # 4 slices + one exact 2 MiB range

    def test_failure_raises_and_reports_partial_completion(self):
        self.native.fail_after = 1
        with self.assertRaisesRegex(RuntimeError, 'DMA-BUF PDMA failed') as cm:
            self.b.write(self.r, 0, length=32 * MiB)
        self.assertIn("'completed': 1", str(cm.exception))

    def test_short_completion_is_rejected(self):
        self.native.run = lambda *a: (0, 0)
        with self.assertRaisesRegex(RuntimeError, 'invalid completion receipt'):
            self.b.read(0, self.r, length=MiB)

    def test_no_dmabuf_capability_is_explicit(self):
        with self.assertRaisesRegex(PathUnavailable, 'NPU_PDMA_CAPS_DMABUF'):
            RngdDmabufBackend(None, native=FakeNative(self.g, open_rc=-95))
        with self.assertRaises(OSError):
            RngdDmabufBackend(None, native=FakeNative(self.g, open_rc=-2))

    def test_parameters_and_close(self):
        for kw in ({'queue_depth': 0}, {'queue_depth': 65}, {'descriptors_per_command': 0}):
            with self.assertRaises(ValueError):
                RngdDmabufBackend(None, native=self.native, **kw)
        self.b.close()
        self.assertTrue(self.native.closed)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.b.read(0, self.r, length=MiB)

    def test_library_manifest_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            lib = os.path.join(d, 'librngd_dmabuf.so')
            open(lib, 'wb').write(b'not the built library')
            open(os.path.join(d, 'librngd_dmabuf.json'), 'w').write('{"sha256": "0", "rngd_driver": "2026.4.0"}')
            with self.assertRaisesRegex(RuntimeError, 'manifest'):
                RngdDmabufBackend(lib)


class CudaRegionGuardTests(unittest.TestCase):
    """cuda_region() refuses before exporting anything when the host cannot export CUDA memory as DMA-BUF."""

    def tensor(self):
        return types.SimpleNamespace(is_cuda=True, is_contiguous=lambda: True, device=types.SimpleNamespace(index=0),
                                     data_ptr=lambda: 1 << 30, numel=lambda: MiB, element_size=lambda: 1)

    def version_file(self, text):
        f = tempfile.NamedTemporaryFile('w', suffix='-nvidia-version', delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_proprietary_module_is_rejected_explicitly(self):
        proprietary = self.version_file('NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Release Build\n')
        with mock.patch.object(paths, 'NVIDIA_VERSION_FILE', proprietary), \
                mock.patch.object(gpu, 'cuda_export') as export:
            with self.assertRaisesRegex(paths.ProprietaryModuleError, 'open kernel module'):
                gpu.cuda_region(self.tensor())
            export.assert_not_called()

    def test_open_module_without_dmabuf_attribute_is_rejected(self):
        open_module = self.version_file('NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.178.04\n')
        with mock.patch.object(paths, 'NVIDIA_VERSION_FILE', open_module), \
                mock.patch.object(gpu, 'dmabuf_supported', return_value=False):
            with self.assertRaisesRegex(PathUnavailable, 'DMA_BUF_SUPPORTED = 0'):
                gpu.cuda_region(self.tensor())

    def test_no_driver_and_non_cuda(self):
        with mock.patch.object(paths, 'NVIDIA_VERSION_FILE', '/nonexistent/nvidia/version'):
            with self.assertRaisesRegex(PathUnavailable, 'no NVIDIA kernel driver'):
                gpu.cuda_region(self.tensor())
        with self.assertRaisesRegex(ValueError, 'CUDA tensor'):
            gpu.cuda_region(types.SimpleNamespace(is_cuda=False))


if __name__ == '__main__':
    unittest.main()
