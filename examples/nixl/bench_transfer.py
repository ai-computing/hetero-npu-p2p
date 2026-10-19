"""RNGD -> ATOM transfer-time curve: existing helper path vs the HETERO_NPU NIXL backend.

Run in the Furiosa environment with NIXL importable and NIXL_PLUGIN_DIR pointing at the built
plugin (see docs/nixl-backend.md). Every (method, size) cell is checked byte-exact on the ATOM side:
the destination is filled with a poison byte first, and after the timed repetitions the first
`size` bytes must equal the source pattern and the rest must still be poison.

  python examples/nixl/bench_transfer.py --atom-python /path/to/atom-venv/bin/python \
      --helper build/rngd-pdma --out results/nixl-curve --sizes-mib 104,256,512,1024,2048,4096
"""
import argparse, json, platform, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / 'src')]

from harness import AtomOwner, RngdSource, pattern, rngd_busy, rngd_channels  # noqa: E402
from hetero_p2p import nixl as hn  # noqa: E402

MiB = 1 << 20
POISON = 0x5A


def nixl_agent_for(name, queue_depth):
    from nixl._api import nixl_agent, nixl_agent_config
    agent = nixl_agent(name, nixl_agent_config(backends=[]))
    agent.create_backend(hn.BACKEND, hn.backend_params(queue_depth=queue_depth))
    return agent


def wait_done(agent, handle, poll=None):
    while True:
        state = agent.check_xfer_state(handle)
        if state == 'DONE':
            return
        if state == 'ERR':
            raise RuntimeError('NIXL transfer failed')
        if poll:
            poll()


class NixlPath:
    """One agent per queue depth; RNGD source and ATOM DMA-BUFs registered once."""

    def __init__(self, name, queue_depth, src, atom, device_path):
        self.agent = nixl_agent_for(name, queue_depth)
        self.src_regions = [(a, n, 0) for a, n in src.regions]
        self.dst_regions = [(r['dva'] + r['offset'], r['size'], 1) for r in atom.regions]
        reg_src = [hn.rngd_region(a, n, device_path, 0) for a, n in src.regions]
        reg_dst = [hn.dmabuf_region(r['fd'], r['size'], base=r['dva'] + r['offset'], dev_id=1, offset=r['offset'])
                   for r in atom.regions]
        self.reg = [self.agent.register_memory(reg_src, 'VRAM', [hn.BACKEND]),
                    self.agent.register_memory(reg_dst, 'VRAM', [hn.BACKEND])]

    def handle(self, offset, nbytes):
        s, d = hn.paired_ranges(self.src_regions, self.dst_regions, offset, nbytes)
        return self.agent.initialize_xfer('WRITE', self.agent.get_xfer_descs(s, 'VRAM'),
                                          self.agent.get_xfer_descs(d, 'VRAM'), self.agent.name,
                                          backends=[hn.BACKEND])

    def whole(self, nbytes):
        t0 = time.perf_counter()
        h = self.handle(0, nbytes)
        t1 = time.perf_counter()
        if self.agent.transfer(h) == 'ERR':
            raise RuntimeError('post failed')
        wait_done(self.agent, h)
        t2 = time.perf_counter()
        h.release()
        return dict(seconds=t2 - t1, prep_seconds=t1 - t0)

    def layerwise(self, nbytes, layers):
        """Post one request per layer chunk in order; a decoder could admit layer i at its fence."""
        handles = [self.handle(off, n) for off, n in hn.chunks(nbytes, layers)]
        t1 = time.perf_counter()
        for h in handles:
            if self.agent.transfer(h) == 'ERR':
                raise RuntimeError('post failed')
        ready = [None] * layers
        while None in ready:
            for i, h in enumerate(handles):
                if ready[i] is None:
                    state = self.agent.check_xfer_state(h)
                    if state == 'ERR':
                        raise RuntimeError(f'layer {i} failed')
                    if state == 'DONE':
                        ready[i] = time.perf_counter() - t1
        for h in handles:
            h.release()
        return dict(seconds=max(ready), first_layer_seconds=ready[0], layer_ready_seconds=ready,
                    in_order=ready == sorted(ready))

    def add_host(self, nbytes):
        """Host DRAM destination for the staging leg (RNGD -> host over PDMA)."""
        import mmap
        import numpy as np
        self.host_buf = mmap.mmap(-1, nbytes)
        self.host = np.frombuffer(self.host_buf, dtype=np.uint8)
        self.reg.append(self.agent.register_memory([hn.host_region(self.host.ctypes.data, nbytes)], 'DRAM', [hn.BACKEND]))

    def to_host(self, nbytes, seed):
        import numpy as np
        self.host[:nbytes] = POISON
        s = hn.map_range(self.src_regions, 0, nbytes)
        d, off = [], 0
        for _, n, _ in s:
            d.append((self.host.ctypes.data + off, n, 0))
            off += n
        h = self.agent.initialize_xfer('WRITE', self.agent.get_xfer_descs(s, 'VRAM'),
                                       self.agent.get_xfer_descs(d, 'DRAM'), self.agent.name, backends=[hn.BACKEND])
        t1 = time.perf_counter()
        if self.agent.transfer(h) == 'ERR':
            raise RuntimeError('post failed')
        wait_done(self.agent, h)
        t2 = time.perf_counter()
        h.release()
        ok = all(np.array_equal(self.host[o:o + n], pattern(o, n, seed))
                 for o in range(0, nbytes, 1 << 30) for n in [min(1 << 30, nbytes - o)])
        return dict(seconds=t2 - t1, host_verified=bool(ok))

    def close(self):
        for r in self.reg:
            self.agent.deregister_memory(r, [hn.BACKEND])


class HelperPath:
    """The existing path: RngdPdmaBackend + native helper, <= 64 MiB per TransferPlan."""

    def __init__(self, helper, device_path, src, atom):
        from hetero_p2p import RngdPdmaBackend
        from hetero_p2p.core import CopySpan, DestinationAllocation, SourceAllocation, TransferPlan
        self.backend = RngdPdmaBackend(helper, device_path)
        self.types = (CopySpan, DestinationAllocation, SourceAllocation, TransferPlan)
        self.src, self.atom, self.device = src, atom, device_path

    def plans(self, nbytes):
        CopySpan, DestinationAllocation, SourceAllocation, TransferPlan = self.types
        src_regions = [(a, n, 0) for a, n in self.src.regions]
        dst_regions = [(r['offset'], r['size'], i) for i, r in enumerate(self.atom.regions)]
        out = []
        for off in range(0, nbytes, 64 * MiB):
            n = min(64 * MiB, nbytes - off)
            s, d = hn.paired_ranges(src_regions, dst_regions, off, n)
            assert len(s) == 1, 'a 64 MiB piece never crosses a 2 GiB region boundary'
            (sa, sn, _), (da, dn, di) = s[0], d[0]
            dest = DestinationAllocation(self.atom.regions[di]['fd'], self.atom, 'raw-bytes')
            source = SourceAllocation(sa, sn, self.src, 'raw-bytes', self.device)
            spans = tuple(CopySpan(0, x, da + x, min(2 * MiB, sn - x)) for x in range(0, sn, 2 * MiB))
            dpc = max(k for k in range(1, 9) if len(spans) % k == 0)
            out.append(TransferPlan(source, (dest,), spans, dpc))
        return out

    def whole(self, nbytes):
        plans = self.plans(nbytes)
        t1 = time.perf_counter()
        commands = 0
        for p in plans:
            receipt = self.backend.transfer(p)
            commands += receipt['completed']
        return dict(seconds=time.perf_counter() - t1, helper_calls=len(plans), commands=commands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atom-python', required=True)
    ap.add_argument('--helper', help='built rngd-pdma (native/build.py); omit to skip the helper path')
    ap.add_argument('--rngd-index', type=int, default=1)
    ap.add_argument('--torch-device', default='furiosa:8', help='logical device of RNGD <rngd-index> PE0')
    ap.add_argument('--sizes-mib', default='104,256,512,1024,2048,4096')
    ap.add_argument('--queue-depths', default='1,4,8')
    ap.add_argument('--layers', type=int, default=32)
    ap.add_argument('--layer-depths', default='', help='queue depths for the layer-wise variant (default: the largest)')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) * MiB for x in a.sizes_mib.split(',')]
    depths = [int(x) for x in a.queue_depths.split(',')]
    device_path = f'/dev/rngd/npu{a.rngd_index}ch0'
    if rngd_busy(a.rngd_index):
        raise SystemExit(f'RNGD npu{a.rngd_index} is in use')
    total = max(sizes)
    seed = 20261019

    t = time.perf_counter()
    src = RngdSource(a.torch_device, total, seed)
    channels = rngd_channels()
    if not channels or any(f'/npu{a.rngd_index}' not in c for c in channels):
        raise SystemExit(f'compute device {a.torch_device} is not RNGD {a.rngd_index}: {channels}')
    src_setup = time.perf_counter() - t
    region_sizes = [n for _, n in src.regions]
    atom = AtomOwner(a.atom_python, out / 'atom', region_sizes)
    rows = []
    result = dict(
        kind='hetero-npu-nixl-curve', started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        source=dict(device='RNGD PE0', torch_device=a.torch_device, regions=region_sizes, setup_seconds=src_setup),
        destination=dict(device='ATOM (rbln:0 of the owner process)', regions=[dict(size=r['size'], export_size=r['export_size']) for r in atom.regions]),
        reps=a.reps, layers=a.layers, poison=POISON, rows=rows,
        environment=dict(python=platform.python_version(), kernel=platform.release()))
    paths = {}
    try:
        if a.helper:
            paths['helper'] = HelperPath(a.helper, device_path, src, atom)
        for qd in depths:
            paths[f'nixl-qd{qd}'] = NixlPath(f'bench-qd{qd}', qd, src, atom, device_path)
        best = paths[f'nixl-qd{max(depths)}']
        methods = [(name, lambda n, p=p: p.whole(n)) for name, p in paths.items()]
        for qd in [int(x) for x in a.layer_depths.split(',')] if a.layer_depths else [max(depths)]:
            methods.append((f'nixl-qd{qd}-layer{a.layers}', lambda n, p=paths[f'nixl-qd{qd}']: p.layerwise(n, a.layers)))
        best.add_host(total)
        for nbytes in sizes:
            # Staging leg for comparison: RNGD -> host DRAM (PDMA), checked on the host.
            reps = [best.to_host(nbytes, seed) for _ in range(a.reps)]
            med = statistics.median(r['seconds'] for r in reps)
            rows.append(dict(method=f'staged-d2h-qd{max(depths)}', bytes=nbytes, seconds=[r['seconds'] for r in reps],
                             median_s=med, gbps=nbytes / med / 1e9, verified=all(r['host_verified'] for r in reps)))
            print(json.dumps({k: rows[-1][k] for k in ('method', 'bytes', 'median_s', 'gbps', 'verified')}), flush=True)
            for name, run in methods:
                fill = atom.call(cmd='fill', value=POISON)
                result.setdefault('atom_fill', []).append(dict(bytes=sum(region_sizes), seconds=fill['seconds']))
                reps = [run(nbytes) for _ in range(a.reps)]
                check = atom.call(cmd='verify', nbytes=nbytes, seed=seed)
                secs = [r['seconds'] for r in reps]
                med = statistics.median(secs)
                row = dict(method=name, bytes=nbytes, seconds=secs, median_s=med,
                           gbps=nbytes / med / 1e9, verified=check['ok'], extra=reps[len(reps) // 2])
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('method', 'bytes', 'median_s', 'gbps', 'verified')}), flush=True)
                (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        result['passed'] = all(r['verified'] for r in rows)
    except Exception as e:
        result.update(passed=False, error=repr(e))
        raise
    finally:
        for name, p in paths.items():
            if hasattr(p, 'close'):
                p.close()
        atom.close()
        result['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
