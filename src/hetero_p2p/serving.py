"""Model-independent configuration and completion validation for replica owners.

An early response is not permission to release or overwrite destination memory.
Only accept a final completion, then finish the scheduler/CopySession leases.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class BatchOptions:
    width: int = 4
    lookahead: int = 8
    collection_seconds: float = .02

    def __post_init__(self):
        if type(self.width) is not int or self.width<1:raise ValueError('positive integer width required')
        if type(self.lookahead) is not int or self.lookahead<self.width:raise ValueError('lookahead must cover one batch')
        if not math.isfinite(self.collection_seconds) or self.collection_seconds<0:raise ValueError('finite nonnegative collection time required')

class CompletionLedger:
    """One active generation per replica; owner-thread use only.

    Reject duplicate, stale, failed, foreign, truncated or changed completions
    before exposing responses. Invalid messages retain the active lease record.
    """
    def __init__(self):self.active={}
    def start(self,replica,batch_id,outputs):
        if replica in self.active:raise RuntimeError('replica already leased')
        if not outputs or any(type(n) is not int or n<1 for n in outputs.values()):raise ValueError('positive output lengths required')
        self.active[replica]={'id':batch_id,'outputs':dict(outputs),'sent':{}}
    def accept(self,replica,message):
        state=self.active.get(replica)
        if state is None or message.get('batch_id')!=state['id']:raise RuntimeError('stale/foreign batch completion')
        if message.get('passed') is not True:raise RuntimeError('decoder validation failed')
        kind=message.get('kind','batch_complete')
        if kind not in ('request_complete','batch_complete'):raise RuntimeError('unknown completion kind')
        rows=message.get('requests');expected=state['outputs']
        if not isinstance(rows,list) or not rows:raise RuntimeError('nonempty completion rows required')
        ids=[r.get('request_id') for r in rows]
        if len(set(ids))!=len(ids) or not set(ids)<=set(expected):raise RuntimeError('duplicate/foreign request')
        final=kind=='batch_complete'
        if final and set(ids)!=set(expected):raise RuntimeError('incomplete batch completion')
        for row in rows:
            rid=row['request_id'];tokens=row.get('tokens')
            if row.get('passed') is not True or not isinstance(tokens,list) or len(tokens)!=expected[rid]:raise RuntimeError('invalid request completion')
            if any(type(t) is not int or t<0 for t in tokens):raise RuntimeError('invalid output token')
            if rid in state['sent']:
                if not final or tokens!=state['sent'][rid]:raise RuntimeError('duplicate or changed response')
        fresh=[row for row in rows if row['request_id'] not in state['sent']]
        for row in fresh:state['sent'][row['request_id']]=list(row['tokens'])
        if final:del self.active[replica]
        return fresh,final

@dataclass(frozen=True)
class RngdBinding:
    """Derive compute and DMA identifiers from one canonical card selection."""
    index: int
    logical_index: int | None = None
    def __post_init__(self):
        if type(self.index) is not int or self.index<0:raise ValueError('nonnegative device index required')
        if self.logical_index is not None and (type(self.logical_index) is not int or self.logical_index<0):raise ValueError('nonnegative logical index required')
    @property
    def device_path(self):return f'/dev/rngd/npu{self.index}ch0'
    @property
    def compute_index(self):return self.index if self.logical_index is None else self.logical_index
    @property
    def torch_device(self):return f'furiosa:{self.compute_index}'
    @classmethod
    def from_path(cls,path,logical_index=None):
        import re
        match=re.fullmatch(r'/dev/rngd/npu(0|[1-9][0-9]*)ch0',path)
        if not match:raise ValueError('canonical RNGD PE0 device path required')
        return cls(int(match.group(1)),logical_index)

    def verify_open_channels(self,paths):
        import re
        observed={int(m.group(1)) for path in paths if (m:=re.fullmatch(r'/dev/rngd/npu([0-9]+)ch[0-7]r?',path))}
        if observed!={self.index}:raise RuntimeError(f'compute/DMA physical mapping mismatch: expected {self.index}, observed {observed}')
