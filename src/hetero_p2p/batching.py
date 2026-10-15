"""Owner-thread scheduling for fixed-width decoder replicas.

This module does not copy memory. Call mark_admitted only after every DMA CQE.
A compiled decoder touches every slot, so a batch cannot run while another
reserved slot on that replica is waiting for admission or its decode request.
"""
from dataclasses import dataclass
import threading


@dataclass
class Reservation:
    request_id: str
    replica: int
    slot: int
    payload: object
    state: str = 'admitting'
    output_tokens: int | None = None
    queued_at: float | None = None


@dataclass(frozen=True)
class Batch:
    replica: int
    generation: int
    requests: tuple[tuple[int, str], ...]


class ReplicaBatchScheduler:
    def __init__(self, replicas, width=4, max_wait_seconds=.01):
        if not replicas or len(set(replicas))!=len(replicas):
            raise ValueError('distinct replicas required')
        if type(width) is not int or width<1 or max_wait_seconds<0:
            raise ValueError('positive width and nonnegative wait required')
        self.replicas=tuple(replicas);self.width=width;self.max_wait=max_wait_seconds
        self.slots={r:{} for r in replicas};self.running={r:None for r in replicas}
        self.generations={r:0 for r in replicas};self.poisoned=set();self.requests={}
        self.owner=threading.get_ident()

    def _check(self):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('scheduler must run on its owning thread')

    def reserve(self, request_id, payload):
        self._check()
        if not isinstance(request_id,str) or not request_id or request_id in self.requests:
            raise ValueError('new nonempty request ID required')
        candidates=[r for r in self.replicas if r not in self.poisoned
                    and self.running[r] is None and len(self.slots[r])<self.width]
        if not candidates:return None
        # Fill an existing batch before starting another replica's batch.
        replica=max(candidates,key=lambda r:len(self.slots[r]))
        slot=next(i for i in range(self.width) if i not in self.slots[replica])
        entry=Reservation(request_id,replica,slot,payload)
        self.slots[replica][slot]=entry;self.requests[request_id]=entry
        return entry

    def mark_admitted(self, request_id):
        self._check();entry=self.requests[request_id]
        if entry.replica in self.poisoned or entry.state!='admitting':
            raise RuntimeError('matching fenced admission required')
        entry.state='ready'

    def queue_decode(self, request_id, output_tokens, now):
        self._check();entry=self.requests[request_id]
        if entry.replica in self.poisoned or entry.state!='ready':
            raise RuntimeError('decode requires completed admission')
        if type(output_tokens) is not int or output_tokens<1:
            raise ValueError('positive output length required')
        entry.output_tokens=output_tokens;entry.queued_at=now;entry.state='queued'

    def dispatchable(self, now):
        self._check();batches=[]
        for replica in self.replicas:
            if replica in self.poisoned or self.running[replica] is not None:continue
            entries=list(self.slots[replica].values())
            if not entries or any(e.state!='queued' for e in entries):continue
            oldest=min(e.queued_at for e in entries)
            if len(entries)<self.width and now-oldest<self.max_wait:continue
            self.generations[replica]+=1
            batch=Batch(replica,self.generations[replica],tuple((e.slot,e.request_id) for e in entries))
            for entry in entries:entry.state='decoding'
            self.running[replica]=batch;batches.append(batch)
        return batches

    def finish(self, batch, passed):
        self._check()
        if self.running.get(batch.replica) is not batch:
            raise RuntimeError('stale or foreign batch completion')
        if passed is not True:
            self.poison(batch.replica)
            raise RuntimeError('decode failed; replica quarantined')
        entries=[self.requests[rid] for _,rid in batch.requests]
        for entry in entries:
            if entry.state!='decoding':raise RuntimeError('invalid slot state')
        for entry in entries:
            del self.requests[entry.request_id];del self.slots[entry.replica][entry.slot]
        self.running[batch.replica]=None
        return entries

    def discard(self, request_id):
        self._check();entry=self.requests[request_id]
        if (entry.replica in self.poisoned or self.running[entry.replica] is not None
                or entry.state!='ready'):
            raise RuntimeError('only fenced, unqueued admission can be discarded')
        del self.requests[request_id];del self.slots[entry.replica][entry.slot]
        return entry

    def poison(self, replica):
        self._check()
        if replica not in self.slots:raise ValueError('unknown replica')
        self.poisoned.add(replica)
        # Retain reservations and their payload owners. Uncertain buffers are
        # never recycled by a timeout or by a failed completion.
        for entry in self.slots[replica].values():entry.state='poisoned'
