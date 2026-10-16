import unittest
from hetero_p2p.iteration import IterationScheduler

class IterationTests(unittest.TestCase):
 def add(self,s,r,n):
  row=s.reserve(r,n,object());s.mark_admitted(r);return row
 def test_refill_only_after_first_completion_preserves_other_sequence(self):
  s=IterationScheduler(2);a=self.add(s,'a',2);b=self.add(s,'b',5)
  t=s.begin(16);self.assertEqual(t.steps,2)
  with self.assertRaises(RuntimeError):s.reserve('c',2,object())
  self.assertEqual(s.complete(t,device_completed=True,passed=True),[a])
  self.assertEqual(b.produced,2);c=self.add(s,'c',3);self.assertEqual(c.slot,a.slot)
  t=s.begin(16);self.assertEqual(t.steps,3)
  self.assertEqual({r.request_id for r in s.complete(t,device_completed=True,passed=True)},{'b','c'})
  self.assertIsNone(s.begin())
 def test_missing_dma_completion_blocks_launch(self):
  s=IterationScheduler(1);s.reserve('a',2,object())
  with self.assertRaises(RuntimeError):s.begin()
 def test_stale_completion_cannot_release_new_owner(self):
  s=IterationScheduler(1);self.add(s,'a',1);t=s.begin();s.complete(t,device_completed=True,passed=True)
  new=self.add(s,'b',1);current=s.begin()
  with self.assertRaises(RuntimeError):s.complete(t,device_completed=True,passed=True)
  self.assertIs(s.slots[0],new);self.assertIs(s.active,current)
 def test_uncertain_completion_retains_owners_and_poison(self):
  s=IterationScheduler(1);row=self.add(s,'a',1);t=s.begin()
  with self.assertRaises(RuntimeError):s.complete(t,device_completed=False,passed=True)
  self.assertIs(s.slots[0],row)
  with self.assertRaises(RuntimeError):s.reserve('b',1,object())
 def test_quantum_yields_without_retiring_incomplete_requests(self):
  s=IterationScheduler(1);row=self.add(s,'a',5)
  for expected in [2,4]:
   t=s.begin(2);self.assertEqual(s.complete(t,device_completed=True,passed=True),[]);self.assertEqual(row.produced,expected)
  t=s.begin(2);self.assertEqual(t.steps,1);self.assertEqual(s.complete(t,device_completed=True,passed=True),[row])
