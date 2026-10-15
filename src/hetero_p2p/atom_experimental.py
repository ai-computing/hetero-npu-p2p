"""Pinned ABI adapter, explicitly experimental: live trace, never stale addresses."""
import ctypes as C
import hashlib,os
from pathlib import Path
from .backend import THUNK_SHA256
from .core import DestinationAllocation


class ExportedCaches:
    def __init__(self, owner, records, mappings, format):
        self.owner,self.records,self.mappings,self.format=owner,records,mappings,format
        self.closed=False

    @property
    def destinations(self):
        if self.closed:raise RuntimeError('exports are closed')
        return tuple(DestinationAllocation(r['fd'],self,self.format) for r in self.records)

    def close(self):
        """Caller must first fence DMA and stop decode; no automatic eviction."""
        if not self.closed:
            for fd in {r['fd'] for r in self.records}:os.close(fd)
            self.closed=True


def export_live_caches(owner, device_addresses, bytes_per_cache, capture,
                       *, experimental=False, format='dlfloat16-token-major-v1'):
    if not experimental:raise RuntimeError('Live-trace ATOM exporter requires explicit experimental opt-in')
    if owner is None or capture.get('pid')!=os.getpid() or bytes_per_cache<=0:
        raise ValueError('capture must belong to this live owning process')
    path=Path('/usr/lib/librbln-thunk.so.3.0.0')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=THUNK_SHA256:
        raise RuntimeError('unvalidated ATOM thunk ABI')
    mappings=[];blocks={}
    for i,addresses in enumerate(device_addresses):
        if len(addresses)!=1:raise ValueError('multi-address cache requires another adapter')
        addr=addresses[0]
        found=[a for a in capture['allocations'] if a['dva']<=addr and addr+bytes_per_cache<=a['dva']+a['size'] and a['flags']==0x22 and a['context'] in capture['contexts']]
        if len(found)!=1:raise ValueError('cache must belong to one live exportable eager allocation')
        a=found[0];blocks[(a['context'],a['dva'])]=a
        mappings.append(dict(index=i,offset=addr-a['dva'],dva=a['dva']))
    lib=C.CDLL(str(path));callback_type=C.CFUNCTYPE(None,C.c_uint64,C.c_uint64,C.c_int)
    fn=lib.rblnExportMemoryByDva;fn.argtypes=[C.c_void_p,callback_type,C.c_uint64];fn.restype=C.c_int
    records=[]
    @callback_type
    def callback(dva,size,fd):records.append(dict(dva=dva,size=size,fd=fd))
    try:
        for (ctx,dva),allocation in blocks.items():
            before=len(records);rc=fn(ctx,callback,dva)
            if rc!=0 or len(records)!=before+1:raise RuntimeError('export did not return exactly one buffer')
            r=records[-1]
            if r['fd']<0 or r['dva']!=dva or r['size']!=allocation['size'] or os.fstat(r['fd']).st_size!=r['size']:
                raise RuntimeError('export metadata mismatch')
        for m in mappings:
            dva=m.pop('dva')
            indices=[i for i,r in enumerate(records) if r['dva']==dva]
            if len(indices)!=1:raise RuntimeError('ambiguous exported allocation')
            m['block_index']=indices[0]
        return ExportedCaches(owner,records,mappings,format)
    except BaseException:
        for fd in {r['fd'] for r in records if r['fd']>=0}:os.close(fd)
        raise
