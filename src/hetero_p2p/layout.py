"""Explicit physical KV layout planning. Conversion belongs to the model adapter."""
from dataclasses import dataclass
from math import gcd
from .core import CopySpan, TransferPlan


@dataclass(frozen=True)
class TokenMajorLayout:
    caches: int
    heads: int
    context: int
    head_dim: int
    element_bytes: int = 2
    format: str = 'dlfloat16-token-major-v1'

    def prefix_plan(self, source, destinations, mappings, prefix_tokens, slot=0, batch=1):
        values=(self.caches,self.heads,self.context,self.head_dim,self.element_bytes,prefix_tokens,batch)
        if any(type(v) is not int or v<=0 for v in values) or type(slot) is not int or not 0<=slot<batch:
            raise ValueError('positive integer dimensions and valid batch slot required')
        if source.format!=self.format or len(mappings)!=self.caches:
            raise ValueError('physical layout/format mismatch')
        head_bytes=self.context*self.head_dim*self.element_bytes
        prefix_bytes=prefix_tokens*self.head_dim*self.element_bytes
        if prefix_tokens>self.context or head_bytes%4096 or prefix_bytes%4096:
            raise ValueError('prefix must already be padded/converted to a 4 KiB head segment')
        if source.size!=self.caches*self.heads*prefix_bytes:
            raise ValueError('source must contain exactly the declared packed prefix payload')
        spans=[]
        for cache,m in enumerate(mappings):
            if m['index']!=cache or m['offset']<0:
                raise ValueError('cache order or allocation offset is invalid')
            for head in range(self.heads):
                spans.append(CopySpan(m['block_index'],(cache*self.heads+head)*prefix_bytes,
                    m['offset']+(slot*self.heads+head)*head_bytes,prefix_bytes))
        return TransferPlan(source,tuple(destinations),tuple(spans),gcd(self.heads,8)).validate()
