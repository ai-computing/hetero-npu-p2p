"""Bounded decode lookahead; selection happens before destination admission."""
from dataclasses import dataclass

@dataclass(frozen=True)
class WaitingDecode:
    request_id: str
    output_tokens: int
    queued_at: float

class LengthQueue:
    def __init__(self, width=4, lookahead=8, wait_seconds=.02):
        if width<1 or lookahead<width or wait_seconds<0:raise ValueError('invalid queue bounds')
        self.width,self.lookahead,self.wait=width,lookahead,wait_seconds
        self.jobs=[]
    def add(self,rid,count,now):
        if not rid or any(j.request_id==rid for j in self.jobs):raise ValueError('duplicate/empty ID')
        if type(count) is not int or count<1:raise ValueError('positive output count required')
        self.jobs.append(WaitingDecode(rid,count,now))
    def take(self,now,*,more_prefills=False,policy='length'):
        if policy not in ('fifo','length'):raise ValueError('unknown scheduling policy')
        if not self.jobs:return []
        # Collect lookahead while the owner already has prefill work, otherwise
        # bound idle collection time. Always include the oldest request.
        if len(self.jobs)<self.lookahead and (more_prefills or now-self.jobs[0].queued_at<self.wait):return []
        anchor=self.jobs[0]
        candidates=self.jobs[:self.lookahead]
        if policy=='length':candidates=sorted(candidates,key=lambda j:(abs(j.output_tokens-anchor.output_tokens),j.queued_at))
        picked=candidates[:self.width];ids={j.request_id for j in picked}
        self.jobs=[j for j in self.jobs if j.request_id not in ids]
        return picked
