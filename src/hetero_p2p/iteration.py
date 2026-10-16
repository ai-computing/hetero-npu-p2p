"""Cooperative iteration scheduling; a worker completion is the only fence.

The caller must establish that every device operation finished before complete().
No timer, early HTTP response, or disconnect may substitute for that proof.
This module performs no DMA and cannot make an asynchronous kernel safe itself.
"""
from dataclasses import dataclass
import threading

@dataclass
class IterationRequest:
    request_id: str
    slot: int
    output_tokens: int
    owner: object
    produced: int = 0
    admitted: bool = False
    tokens: tuple[int, ...] = ()

@dataclass(frozen=True)
class IterationTicket:
    generation: int
    steps: int
    requests: tuple[tuple[int, str], ...]

class IterationScheduler:
    def __init__(self, width):
        if type(width) is not int or width < 1:raise ValueError('positive width required')
        self.width=width;self.slots={};self.active=None;self.generation=0;self.poisoned=False
        self.owner_thread=threading.get_ident()

    def _fenced(self):
        if threading.get_ident()!=self.owner_thread:raise RuntimeError('owner thread required')
        if self.poisoned:raise RuntimeError('uncertain device state; retain buffer owners')
        if self.active is not None:raise RuntimeError('worker completion required before slot mutation')

    def reserve(self, request_id, output_tokens, owner):
        self._fenced()
        if not isinstance(request_id,str) or not request_id or any(r.request_id==request_id for r in self.slots.values()):raise ValueError('distinct nonempty request ID required')
        if type(output_tokens) is not int or output_tokens<1:raise ValueError('positive output length required')
        if len(self.slots)==self.width:return None
        slot=next(i for i in range(self.width) if i not in self.slots)
        row=IterationRequest(request_id,slot,output_tokens,owner);self.slots[slot]=row
        return row

    def mark_admitted(self, request_id):
        self._fenced()
        row=next((r for r in self.slots.values() if r.request_id==request_id),None)
        if row is None or row.admitted:raise ValueError('matching new admission required')
        row.admitted=True

    def begin(self, quantum=1):
        self._fenced()
        if type(quantum) is not int or quantum<1:raise ValueError('positive quantum required')
        if not self.slots:return None
        if any(not r.admitted for r in self.slots.values()):raise RuntimeError('DMA completion required for every reserved slot')
        # Yield no later than the earliest request completion so it can be replaced.
        steps=min(quantum,min(r.output_tokens-r.produced for r in self.slots.values()))
        self.generation+=1
        ticket=IterationTicket(self.generation,steps,tuple((i,r.request_id) for i,r in sorted(self.slots.items())))
        self.active=ticket
        return ticket

    def complete(self, ticket, *, device_completed, passed):
        if threading.get_ident()!=self.owner_thread:raise RuntimeError('owner thread required')
        if self.poisoned or self.active is not ticket:raise RuntimeError('stale or foreign completion')
        if device_completed is not True or passed is not True:
            self.poisoned=True
            raise RuntimeError('uncertain or failed iteration; retain all allocations')
        finished=[]
        for slot,rid in ticket.requests:
            row=self.slots[slot]
            if row.request_id!=rid:raise RuntimeError('slot ownership changed')
            row.produced+=ticket.steps
            if row.produced==row.output_tokens:finished.append(row)
        for row in finished:del self.slots[row.slot]
        self.active=None
        return finished

    def accept_reply(self, ticket, reply):
        """Validate the worker protocol before allowing any slot mutation."""
        try:
            if self.active is not ticket or reply.get('kind')!='iteration_complete' or reply.get('generation')!=ticket.generation or reply.get('steps')!=ticket.steps:raise RuntimeError('iteration identity mismatch')
            rows=reply['requests']
            if len(rows)!=len(ticket.requests) or {(r['slot'],r['request_id']) for r in rows}!=set(ticket.requests):raise RuntimeError('iteration request set mismatch')
            for row in rows:
                entry=self.slots[row['slot']];expected=entry.produced+ticket.steps
                if row.get('passed') is not True or row.get('reference_checked') is not True or row.get('max_abs_diff')!=0:raise RuntimeError('iteration validation failed')
                if row.get('produced')!=expected or len(row['tokens'])!=expected or row.get('complete') is not (expected==entry.output_tokens):raise RuntimeError('iteration progress mismatch')
                if tuple(row['tokens'][:entry.produced])!=entry.tokens:raise RuntimeError('previous output changed')
            entries=dict(self.slots)
            finished=self.complete(ticket,device_completed=reply.get('device_completed'),passed=reply.get('passed'))
            for row in rows:
                entries[row['slot']].tokens=tuple(row['tokens'])
            return finished
        except BaseException:
            self.poisoned=True
            raise

@dataclass(frozen=True)
class QuantumPolicy:
    """Bound admission delay when slots are empty; amortize control when full.

    This is a work bound, not a wall-clock SLO guarantee. Device speed, source
    preparation and admission must be measured separately.
    """
    full: int = 128
    partial: int = 4

    def __post_init__(self):
        if any(type(n) is not int or not 1<=n<=128 for n in (self.full,self.partial)):
            raise ValueError('quantum must be an integer in [1,128]')
        if self.partial>self.full:raise ValueError('partial quantum exceeds full quantum')

    def choose(self, occupied, width):
        if type(width) is not int or type(occupied) is not int or not 1<=occupied<=width:
            raise ValueError('positive occupancy within batch width required')
        return self.full if occupied==width else self.partial
