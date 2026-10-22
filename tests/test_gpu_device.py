"""GPU <-> RNGD on real devices. Skipped unless HETERO_P2P_DMABUF_LIB names the built library, the RNGD in
HETERO_P2P_DEVICE (default /dev/rngd/npu0ch0) exists, and torch sees a GPU. Writes RNGD DRAM at
HETERO_P2P_DMABUF_NPU_OFFSET (default 4 GiB) without an allocation: run it only on an idle chip."""
import os, unittest

LIB = os.environ.get('HETERO_P2P_DMABUF_LIB', '')
DEVICE = os.environ.get('HETERO_P2P_DEVICE', '/dev/rngd/npu0ch0')
NPU = int(os.environ.get('HETERO_P2P_DMABUF_NPU_OFFSET', str(4 << 30)))
try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False
MiB = 1024 ** 2


@unittest.skipUnless(LIB and os.path.exists(LIB) and os.path.exists(DEVICE) and CUDA, 'needs GPU, RNGD and the library')
class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hetero_p2p.gpu import RngdDmabufBackend
        cls.backend = RngdDmabufBackend(LIB, DEVICE, queue_depth=4)

    @classmethod
    def tearDownClass(cls):
        cls.backend.close()

    def test_round_trip_through_the_api(self):
        from hetero_p2p.gpu import cuda_region, release_region
        src = torch.randint(0, 256, (64 * MiB,), dtype=torch.uint8, device='cuda')
        dst = torch.zeros_like(src)
        try:
            for n in (2 * MiB, 64 * MiB):
                w = self.backend.write(src, NPU, length=n)
                r = self.backend.read(NPU, dst, length=n)
                self.assertEqual((w['completed'], r['completed']), (w['requested'], r['requested']))
                self.assertTrue(torch.equal(src[:n], dst[:n]), n)
            self.assertIs(cuda_region(src), cuda_region(src))             # registered once
        finally:
            release_region(src)
            release_region(dst)

    def test_many_commands_at_every_queue_depth(self):
        from hetero_p2p.gpu import RngdDmabufBackend, release_region
        src = torch.randint(0, 256, (16 * MiB,), dtype=torch.uint8, device='cuda')
        dst = torch.zeros_like(src)
        try:
            for qd in (1, 2, 3, 8):
                b = RngdDmabufBackend(LIB, DEVICE, queue_depth=qd, descriptors_per_command=1)
                try:
                    w = b.write(src, NPU)                       # 8 commands of one 2 MiB descriptor
                    dst.zero_()
                    r = b.read(NPU, dst)
                finally:
                    b.close()
                self.assertEqual((w['requested'], w['completed'], r['completed']), (8, 8, 8))
                self.assertTrue(torch.equal(src, dst), qd)
        finally:
            release_region(src)
            release_region(dst)

    def test_invalid_descriptors_rejected_before_submission(self):
        from hetero_p2p.gpu import cuda_region, release_region
        t = torch.zeros(2 * MiB, dtype=torch.uint8, device='cuda')
        try:
            fd = cuda_region(t).range_fd(0, 2 * MiB)
            native, ctx = self.backend._native, self.backend._ctx
            for npu, off, n in ((NPU + 1, 0, 4096), (NPU, 4096, 2 * MiB), ((48 << 30), 0, 4096)):
                rc, done = native.run(ctx, True, [fd], [1], [npu], [off], [n])
                self.assertEqual((rc, done), (-22, 0))
            self.backend.write(t, NPU)                                     # the context is still usable
        finally:
            release_region(t)


if __name__ == '__main__':
    unittest.main()
