"""Synchronous decode quanta with per-slot trajectories and explicit device fence.

The model adapter supplies step_batch, notify_cache_write and synchronize.
Reference logits remain mandatory in this experimental validation path.
"""
import time
import numpy as np

class IterationDecoder:
    def __init__(self, engine, synchronize, references):
        self.engine=engine;self.synchronize=synchronize;self.references=references
        self.slots={};self.generation=0;self.poisoned=False

    def run(self, command):
        if self.poisoned:raise RuntimeError('decoder quarantined')
        try:return self._run(command)
        except BaseException:
            self.poisoned=True
            raise

    def _run(self, command):
        generation=command['generation'];steps=command['steps'];jobs=command['requests']
        if type(generation) is not int or generation!=self.generation+1:raise ValueError('unexpected generation')
        if type(steps) is not int or steps<1:raise ValueError('positive step count required')
        if not 1<=len(jobs)<=self.engine.batch_size:raise ValueError('invalid batch width')
        if len({j['slot'] for j in jobs})!=len(jobs) or len({j['id'] for j in jobs})!=len(jobs):raise ValueError('duplicate slot/request')
        incoming={j['slot']:j for j in jobs}
        if any(slot not in incoming for slot in self.slots):raise ValueError('active trajectory omitted')
        new=[]
        for j in jobs:
            slot=j['slot'];tokens=j['tokens'];count=j['output'];produced=j['produced']
            if type(slot) is not int or not 0<=slot<self.engine.batch_size:raise ValueError('invalid slot')
            if type(count) is not int or not 1<=count<=128 or type(produced) is not int or not 0<=produced<count:raise ValueError('invalid progress')
            if count-produced<steps or len(tokens)+count>getattr(self.engine,'context',512):raise ValueError('quantum exceeds request/context')
            key=tuple(tokens)
            if key not in self.references:raise ValueError('full-logit reference required')
            if slot in self.slots:
                s=self.slots[slot]
                if s['id']!=j['id'] or s['prompt']!=key or s['output']!=count or len(s['tokens'])!=produced:raise ValueError('trajectory mismatch')
            else:
                if produced!=0 or j.get('admitted') is not True:raise ValueError('new slot requires fenced admission')
                new.append((slot,j,key))
        # The producer admits only after the preceding synchronized reply.
        if new:self.engine.notify_cache_write(range(64))
        for slot,j,key in new:
            self.slots[slot]=dict(id=j['id'],prompt=key,output=j['output'],tokens=[],current=key[-1],start=None,first=None)
        start=time.perf_counter()
        for _ in range(steps):
            current=[128000]*self.engine.batch_size;positions=[0]*self.engine.batch_size
            for slot,s in self.slots.items():
                current[slot]=s['current'];positions[slot]=len(s['prompt'])-1+len(s['tokens'])
                if s['start'] is None:s['start']=time.perf_counter()
            out=self.engine.step_batch(current,positions)
            now=time.perf_counter()
            for slot,s in self.slots.items():
                row=out[slot];index=len(s['tokens']);expected=self.references[s['prompt']][index]
                if not np.isfinite(row).all() or not np.array_equal(row,expected):raise RuntimeError('iteration full-logit parity failure')
                s['current']=int(row.argmax());s['tokens'].append(s['current'])
                if s['first'] is None:s['first']=now
        # step_batch already materializes CPU logits; explicit synchronize also
        # fences any adapter side effects before authorizing external writes.
        self.synchronize();end=time.perf_counter();rows=[];finished=[]
        for slot,s in self.slots.items():
            complete=len(s['tokens'])==s['output']
            rows.append(dict(request_id=s['id'],slot=slot,produced=len(s['tokens']),complete=complete,tokens=list(s['tokens']),passed=True,reference_checked=True,max_abs_diff=0.0,start=s['start'],first=s['first'],end=end))
            if complete:finished.append(slot)
        for slot in finished:del self.slots[slot]
        self.generation=generation
        return dict(kind='iteration_complete',generation=generation,steps=steps,device_completed=True,passed=True,requests=rows,seconds=end-start)
