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

class QuantumPolicyTests(unittest.TestCase):
 def test_partial_is_bounded_and_full_is_amortized(self):
  from hetero_p2p.iteration import QuantumPolicy
  q=QuantumPolicy(128,4)
  self.assertEqual([q.choose(n,8) for n in range(1,9)],[4]*7+[128])
 def test_fixed_controls_and_invalid_configuration(self):
  from hetero_p2p.iteration import QuantumPolicy
  for n in [4,128]:
   q=QuantumPolicy(n,n);self.assertEqual(q.choose(1,8),n);self.assertEqual(q.choose(8,8),n)
  for args in [(4,128),(0,1),(128,False),(129,4)]:
   with self.assertRaises(ValueError):QuantumPolicy(*args)
  for used,width in [(0,8),(9,8),(1,0),(True,8)]:
   with self.assertRaises(ValueError):QuantumPolicy().choose(used,width)
