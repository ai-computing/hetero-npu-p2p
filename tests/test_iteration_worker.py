import unittest
import numpy as np
from hetero_p2p.iteration_worker import IterationDecoder
from hetero_p2p.iteration import IterationScheduler

class FakeEngine:
 batch_size=2
 def __init__(self):self.fenced=True;self.notifications=0
 def notify_cache_write(self,indices):
  assert self.fenced;assert list(indices)==list(range(64));self.notifications+=1
 def step_batch(self,current,positions):
  self.fenced=False
  return np.array([[float(p),0,-1] for p in positions])
 def synchronize(self):self.fenced=True

class IterationWorkerTests(unittest.TestCase):
 def setup_decoder(self):
  e=FakeEngine();refs={(1,2):np.array([[float(i+1),0,-1] for i in range(128)])}
  return e,IterationDecoder(e,e.synchronize,refs),IterationScheduler(2)
 def add(self,s,r,count):
  row=s.reserve(r,count,(1,2));s.mark_admitted(r);return row
 def run_quantum(self,d,s,quantum):
  t=s.begin(quantum);cmd=dict(generation=t.generation,steps=t.steps,requests=[dict(slot=slot,id=rid,tokens=[1,2],output=s.slots[slot].output_tokens,produced=s.slots[slot].produced,admitted=True) for slot,rid in t.requests])
  reply=d.run(cmd);finished=s.accept_reply(t,reply);return reply,finished
 def test_refill_keeps_continuing_trajectory_and_fences(self):
  e,d,s=self.setup_decoder();a=self.add(s,'a',2);b=self.add(s,'b',5)
  reply,done=self.run_quantum(d,s,128);self.assertEqual(done,[a]);self.assertEqual(a.tokens,(0,0));self.assertTrue(e.fenced);self.assertEqual(b.tokens,(0,0))
  c=self.add(s,'c',3);self.assertEqual(c.slot,a.slot)
  reply,done=self.run_quantum(d,s,128);self.assertEqual({r.request_id for r in done},{'b','c'});self.assertEqual(e.notifications,2);self.assertEqual(b.tokens,(0,)*5);self.assertEqual(c.tokens,(0,)*3);self.assertFalse(d.slots)
 def test_forged_progress_poison_retains_owner(self):
  e,d,s=self.setup_decoder();row=self.add(s,'a',2);t=s.begin(1)
  reply=dict(kind='iteration_complete',generation=t.generation,steps=1,requests=[dict(slot=0,request_id='a',produced=2,tokens=[0,0],complete=True,passed=True,reference_checked=True,max_abs_diff=0)],device_completed=True,passed=True)
  with self.assertRaises(RuntimeError):s.accept_reply(t,reply)
  self.assertTrue(s.poisoned);self.assertIs(s.slots[0],row)
 def test_worker_rejects_stale_generation_before_execution(self):
  e,d,s=self.setup_decoder();self.add(s,'a',2);self.run_quantum(d,s,1)
  with self.assertRaises(ValueError):d.run(dict(generation=1,steps=1,requests=[]))
  self.assertTrue(d.poisoned)
