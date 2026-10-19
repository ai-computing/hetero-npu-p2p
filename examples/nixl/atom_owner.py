"""ATOM destination owner for the NIXL test/benchmark. Run under examples/koni/capture.gdb.

Allocates one eager uint8 tensor per region, exports each live allocation as a DMA-BUF
(hetero_p2p.atom_experimental, pinned thunk ABI), sends the fds to the RNGD process and answers:
  {"cmd": "fill", "value": v}                  write v to every region (host -> ATOM, SDK path)
  {"cmd": "verify", "nbytes": n, "seed": s}    first n logical bytes == pattern(s), rest == fill value
  {"cmd": "stop"}
"""
import array, ctypes, json, os, socket, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch_rbln  # noqa: F401  registers the rbln device
import rebel

from hetero_p2p.atom_experimental import export_live_caches
from harness import pattern

config = json.loads(Path(sys.argv[1]).read_text())
out = Path(config['out'])
vmem = rebel._C.vmem
tensors, exports, fill_value = [], [], None
conn = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
try:
    regions = []
    for size in config['regions']:
        t = torch.zeros(size, dtype=torch.uint8).to('rbln:0')
        tensors.append(t)
    capture = json.loads((out / 'live-capture.json').read_text())
    for i, t in enumerate(tensors):
        ex = export_live_caches(t, [vmem.get_device_addrs(t.data_ptr())], t.numel(), capture, experimental=True)
        exports.append(ex)
        rec, m = ex.records[0], ex.mappings[0]
        regions.append(dict(block=i, export_size=rec['size'], offset=m['offset'], dva=rec['dva'], size=t.numel()))
    # Stop tracing before the measured phase (capture.gdb disables breakpoints on getppid).
    (out / 'capture-complete').write_text('exports done')
    ctypes.CDLL(None).getppid()
    conn.connect(config['socket'])
    fds = [ex.records[0]['fd'] for ex in exports]
    conn.sendmsg([json.dumps(dict(pid=os.getpid(), regions=regions)).encode()],
                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', fds))])
    while True:
        cmd = json.loads(conn.recv(65536))
        try:
            if cmd['cmd'] == 'stop':
                conn.send(json.dumps(dict(ok=True)).encode())
                break
            if cmd['cmd'] == 'fill':
                start = time.perf_counter()
                fill_value = int(cmd['value'])
                for t in tensors:
                    t.copy_(torch.full((t.numel(),), fill_value, dtype=torch.uint8))
                    vmem.sync_to_device(t.data_ptr())
                reply = dict(ok=True, seconds=time.perf_counter() - start)
            elif cmd['cmd'] == 'verify':
                start, nbytes, seed = time.perf_counter(), int(cmd['nbytes']), int(cmd['seed'])
                bad, offset = [], 0
                for i, t in enumerate(tensors):
                    # The DMA wrote device memory behind the runtime: drop any host-side copy first.
                    vmem.mark_device_memory_is_updated(t.data_ptr())
                    got = t.cpu().numpy()
                    k = max(0, min(t.numel(), nbytes - offset))
                    if k and not np.array_equal(got[:k], pattern(offset, k, seed)):
                        bad.append(dict(region=i, kind='payload', first=int(np.argmax(got[:k] != pattern(offset, k, seed)))))
                    if k < t.numel() and fill_value is not None and not np.all(got[k:] == fill_value):
                        bad.append(dict(region=i, kind='beyond-range', first=k + int(np.argmax(got[k:] != fill_value))))
                    offset += t.numel()
                reply = dict(ok=not bad, mismatches=bad, seconds=time.perf_counter() - start)
            else:
                reply = dict(ok=False, error=f'unknown command {cmd}')
        except Exception as e:
            reply = dict(ok=False, error=repr(e), traceback=traceback.format_exc())
        conn.send(json.dumps(reply).encode())
except Exception:
    traceback.print_exc()
    raise
finally:
    conn.close()
    for ex in exports:
        ex.close()
