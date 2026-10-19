"""Hardware test of the HETERO_NPU NIXL backend through the NIXL Python API (nixl_agent):
register -> initialize_xfer -> transfer -> check_xfer_state -> verify bytes.

Skipped unless HETERO_NPU_NIXL=1. Needs, in the Furiosa environment: NIXL importable, the built
plugin in NIXL_PLUGIN_DIR, an idle RNGD. Optional HETERO_NPU_ATOM_PYTHON (ATOM environment's
python) adds the RNGD -> ATOM DMA-BUF case. Other settings: HETERO_NPU_RNGD_INDEX (1),
HETERO_NPU_TORCH_DEVICE (furiosa:8, the logical device of that RNGD's PE0), HETERO_NPU_OUT.
"""
import mmap, os, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENABLED = os.environ.get('HETERO_NPU_NIXL') == '1'
NBYTES = 8 << 20


def host_buffer(nbytes):
    import numpy as np
    buf = mmap.mmap(-1, nbytes)  # page aligned, as the PDMA scatter path requires
    return buf, np.frombuffer(buf, dtype=np.uint8)


@unittest.skipUnless(ENABLED, 'hardware test: set HETERO_NPU_NIXL=1')
class HeteroNpuNixlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(ROOT / 'examples' / 'nixl'), str(ROOT / 'src')]
        import torch
        from nixl._api import nixl_agent, nixl_agent_config
        from harness import RngdSource, rngd_busy
        from hetero_p2p import nixl as hn
        cls.hn, cls.torch = hn, torch
        cls.index = int(os.environ.get('HETERO_NPU_RNGD_INDEX', '1'))
        cls.device_path = f'/dev/rngd/npu{cls.index}ch0'
        cls.torch_device = os.environ.get('HETERO_NPU_TORCH_DEVICE', 'furiosa:8')
        if rngd_busy(cls.index):
            raise unittest.SkipTest(f'RNGD npu{cls.index} is in use')
        cls.seed = 7
        cls.src = RngdSource(cls.torch_device, NBYTES, cls.seed)      # pattern-filled
        cls.scratch = RngdSource(cls.torch_device, NBYTES, cls.seed + 1)
        cls.agent = nixl_agent('hetero-npu-test', nixl_agent_config(backends=[]))
        cls.agent.create_backend(hn.BACKEND, hn.backend_params(queue_depth=2))
        (addr, n), = cls.src.regions
        (saddr, sn), = cls.scratch.regions
        cls.agent.register_memory([hn.rngd_region(addr, n, cls.device_path, 0),
                                   hn.rngd_region(saddr, sn, cls.device_path, 1)], 'VRAM', [hn.BACKEND])

    def wait(self, handle):
        while True:
            state = self.agent.check_xfer_state(handle)
            if state != 'PROC':
                return state

    def xfer(self, op, local, local_type, remote, remote_type, notif=b''):
        a = self.agent
        h = a.initialize_xfer(op, a.get_xfer_descs(local, local_type), a.get_xfer_descs(remote, remote_type),
                              a.name, notif, backends=[self.hn.BACKEND])
        self.assertIn(a.transfer(h), ('PROC', 'DONE'))
        state = self.wait(h)
        h.release()
        return state

    def test_host_rngd_round_trip(self):
        import numpy as np
        from harness import pattern
        hn = self.hn
        _a, a = host_buffer(NBYTES)
        _b, b = host_buffer(NBYTES)
        a[:] = pattern(0, NBYTES, 99)
        reg = self.agent.register_memory([hn.host_region(a.ctypes.data, NBYTES), hn.host_region(b.ctypes.data, NBYTES)],
                                         'DRAM', [hn.BACKEND])
        try:
            (saddr, _), = self.scratch.regions
            # host -> RNGD (SCATTER_WRITE), then RNGD -> another host buffer (SCATTER_READ).
            self.assertEqual(self.xfer('WRITE', [(a.ctypes.data, NBYTES, 0)], 'DRAM', [(saddr, NBYTES, 1)], 'VRAM'), 'DONE')
            self.assertEqual(self.xfer('READ', [(b.ctypes.data, NBYTES, 0)], 'DRAM', [(saddr, NBYTES, 1)], 'VRAM'), 'DONE')
            self.assertTrue(np.array_equal(a, b))
            # Independent check through the Furiosa SDK: the device now holds the host pattern.
            self.assertTrue(np.array_equal(self.scratch.tensors[0].cpu().numpy(), a))
        finally:
            self.agent.deregister_memory(reg, [hn.BACKEND])

    def test_unvalidated_routes_are_refused(self):
        hn = self.hn
        _a, a = host_buffer(NBYTES)
        _b, b = host_buffer(NBYTES)
        reg = self.agent.register_memory([hn.host_region(a.ctypes.data, NBYTES), hn.host_region(b.ctypes.data, NBYTES)],
                                         'DRAM', [hn.BACKEND])
        try:
            (addr, _), = self.src.regions
            (saddr, _), = self.scratch.regions
            with self.assertRaises(Exception):  # host -> host is not a PDMA route
                self.xfer('WRITE', [(a.ctypes.data, 4096, 0)], 'DRAM', [(b.ctypes.data, 4096, 0)], 'DRAM')
            with self.assertRaises(Exception):  # RNGD -> RNGD is not validated
                self.xfer('WRITE', [(addr, 4096, 0)], 'VRAM', [(saddr, 4096, 1)], 'VRAM')
            with self.assertRaises(Exception):  # unaligned
                self.xfer('WRITE', [(a.ctypes.data + 1, 4096, 0)], 'DRAM', [(saddr, 4096, 1)], 'VRAM')
        finally:
            self.agent.deregister_memory(reg, [hn.BACKEND])

    @unittest.skipUnless(os.environ.get('HETERO_NPU_ATOM_PYTHON'), 'set HETERO_NPU_ATOM_PYTHON for RNGD -> ATOM')
    def test_rngd_to_atom_dmabuf(self):
        from harness import AtomOwner
        hn = self.hn
        out = Path(os.environ.get('HETERO_NPU_OUT') or tempfile.mkdtemp(prefix='hetero-npu-nixl-'))
        atom = AtomOwner(os.environ['HETERO_NPU_ATOM_PYTHON'], out, [NBYTES])
        try:
            r, = atom.regions
            base = r['dva'] + r['offset']
            reg = self.agent.register_memory([hn.dmabuf_region(r['fd'], r['size'], base, 2, r['offset'])], 'VRAM', [hn.BACKEND])
            try:
                (addr, _), = self.src.regions
                half = NBYTES // 2
                atom.call(cmd='fill', value=0x5A)
                # Half the buffer: the verify also proves nothing beyond the range was written.
                state = self.xfer('WRITE', [(addr, half, 0)], 'VRAM', [(base, half, 2)], 'VRAM', notif=b'kv-ready')
                self.assertEqual(state, 'DONE')
                # NIXL lists only remote-capable backends by default: name the local one.
                notifs = self.agent.get_new_notifs([hn.BACKEND])
                self.assertIn(b'kv-ready', notifs.get(self.agent.name, []))
                self.assertTrue(atom.call(cmd='verify', nbytes=half, seed=self.seed)['ok'])
                with self.assertRaises(Exception):  # DMA-BUF -> RNGD (reverse) is refused
                    self.xfer('READ', [(addr, 4096, 0)], 'VRAM', [(base, 4096, 2)], 'VRAM')
            finally:
                self.agent.deregister_memory(reg, [hn.BACKEND])
        finally:
            atom.close()


if __name__ == '__main__':
    unittest.main()
