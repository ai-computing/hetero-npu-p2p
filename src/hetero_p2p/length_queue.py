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
    def take(self,now,*,more_prefills=False,policy='length',limit=None):
        if policy not in ('fifo','length','padding'):raise ValueError('unknown scheduling policy')
        limit=self.width if limit is None else limit
        if type(limit) is not int or not 1<=limit<=self.width:raise ValueError('limit must fit batch width')
        if not self.jobs:return []
        # Collect lookahead while the owner already has prefill work, otherwise
        # bound idle collection time. Always include the oldest request.
        if limit==self.width and len(self.jobs)<self.lookahead and (more_prefills or now-self.jobs[-1].queued_at<self.wait):return []
        anchor=self.jobs[0]
        candidates=self.jobs[:self.lookahead]
        if policy=='length':candidates=sorted(candidates,key=lambda j:(abs(j.output_tokens-anchor.output_tokens),j.queued_at))
        if policy=='padding':
            # Adapt length-aware batching to fixed-width autoregressive decode.
            # Minimize wasted lane-steps while always serving the oldest job.
            # For each possible maximum, fill with the largest eligible jobs.
            size=min(limit,len(candidates));choices=[]
            for maximum in sorted({j.output_tokens for j in candidates if j.output_tokens>=anchor.output_tokens}):
                eligible=sorted((j for j in candidates if j is not anchor and j.output_tokens<=maximum),key=lambda j:(-j.output_tokens,j.queued_at))
                if len(eligible)<size-1:continue
                group=[anchor]+eligible[:size-1]
                cost=self.width*max(j.output_tokens for j in group)-sum(j.output_tokens for j in group)
                choices.append((cost,max(j.output_tokens for j in group),tuple(sorted(j.queued_at for j in group)),group))
            picked=min(choices,key=lambda x:x[:3])[3]
        else:picked=candidates[:limit]
        ids={j.request_id for j in picked}
        self.jobs=[j for j in self.jobs if j.request_id not in ids]
        return picked
