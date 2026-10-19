"""Shared pieces of the NIXL hardware test and transfer benchmark (RNGD process side).

The ATOM destination lives in a separate process (`atom_owner.py`, ATOM Python environment,
started under examples/koni/capture.gdb so the live allocation can be exported). It sends the
exported DMA-BUF fds over a Unix socket and answers fill/verify commands.
"""
import array, json, os, socket, subprocess, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REGION_BYTES = 2 << 30   # both SDKs reject a single tensor of 4 GiB or more
PAGE = 4096


def pattern(offset, nbytes, seed):
    """Deterministic bytes for logical range [offset, offset+nbytes): a seeded 1 MiB block tiled,
    with the first 8 bytes of every 4 KiB page set to (page index, seed) so misplaced pages show."""
    assert offset % PAGE == 0 and nbytes % PAGE == 0
    block = np.random.default_rng(seed).integers(0, 256, 1 << 20, dtype=np.uint8)
    out = np.resize(block, nbytes)
    pages = out.view(np.uint64)[::PAGE // 8]
    pages[:] = np.arange(offset // PAGE, (offset + nbytes) // PAGE, dtype=np.uint64) ^ (np.uint64(seed) << np.uint64(40))
    return out


def rngd_busy(index):
    return f'npu{index}:' in subprocess.check_output(['furiosa-smi', 'ps'], text=True)


def rngd_channels():
    found = set()
    for d in os.listdir('/proc/self/fd'):
        try:
            p = os.readlink(f'/proc/self/fd/{d}')
        except OSError:
            continue
        if p.startswith('/dev/rngd/'):
            found.add(p)
    return sorted(found)


class RngdSource:
    """Pattern-filled flat uint8 RNGD allocations, REGION_BYTES each (logical concatenation)."""

    def __init__(self, torch_device, nbytes, seed):
        import torch
        import furiosa.torch as ft
        from furiosa.torch._C.ir import FlatDramShape
        from furiosa.torch._C.tensor import copy_from_tensor
        self.device, self.tensors, self.regions = torch_device, [], []
        for off in range(0, nbytes, REGION_BYTES):
            n = min(REGION_BYTES, nbytes - off)
            host = torch.from_numpy(pattern(off, n, seed))
            t = copy_from_tensor(host, FlatDramShape.new_as_broadcast([n], torch.uint8), torch_device)
            self.tensors.append(t)
            self.regions.append((t.data_ptr(), n))
        ft.current_stream(torch_device).synchronize()


class AtomOwner:
    """Start atom_owner.py under gdb capture, receive its exported DMA-BUF fds."""

    def __init__(self, atom_python, out_dir, region_sizes, timeout=600):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ('live-capture.json', 'capture-complete', 'breakpoints-disabled'):
            (self.out / name).unlink(missing_ok=True)
        sock_path = str(self.out / 'atom.sock')
        if len(sock_path.encode()) >= 108:
            raise ValueError('socket path too long')
        Path(sock_path).unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(1)
        config = dict(socket=sock_path, out=str(self.out), regions=list(region_sizes))
        (self.out / 'atom-config.json').write_text(json.dumps(config))
        env = dict(os.environ, HETERO_CAPTURE_DIR=str(self.out), TORCH_RBLN_EAGER_MALLOC='1',
                   PYTHONPATH=str(HERE.parent.parent / 'src'))
        for k in ('RBLN_DEVICE_MAP', 'RBLN_DUMMY_DEVICE', 'PYTHONHOME'):
            env.pop(k, None)
        self.log = (self.out / 'atom-owner.log').open('w')
        self.proc = subprocess.Popen(['gdb', '-q', '-batch', '-x', str(HERE.parent / 'koni' / 'capture.gdb'),
                                      '--args', atom_python, str(HERE / 'atom_owner.py'), str(self.out / 'atom-config.json')],
                                     env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f'ATOM owner exited; see {self.out}/atom-owner.log')
            if time.monotonic() > deadline:
                self.proc.kill()
                raise RuntimeError('ATOM owner did not export in time')
            try:
                self.conn, _ = listener.accept()
                break
            except socket.timeout:
                pass
        listener.close()
        data, anc, flags, _ = self.conn.recvmsg(65536, socket.CMSG_SPACE(16 * 4))
        assert not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
        fds = array.array('i')
        for level, kind, payload in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(payload)
        self.meta = json.loads(data)
        # One record per region: fd, export size, byte offset of the tensor in the export, DVA.
        self.regions = []
        for r in self.meta['regions']:
            fd = fds[r['block']]
            assert os.fstat(fd).st_size == r['export_size']
            self.regions.append(dict(r, fd=fd))

    def call(self, **cmd):
        self.conn.send(json.dumps(cmd).encode())
        reply = json.loads(self.conn.recv(65536))
        if not reply.get('ok', False):
            raise RuntimeError(f'ATOM owner: {reply}')
        return reply

    def close(self):
        try:
            self.call(cmd='stop')
        except Exception:
            pass
        self.conn.close()
        for fd in {r['fd'] for r in self.regions}:
            os.close(fd)
        try:
            self.proc.wait(timeout=120)
        finally:
            self.log.close()
