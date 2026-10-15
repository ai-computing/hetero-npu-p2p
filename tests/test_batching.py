import threading,unittest
from hetero_p2p.batching import ReplicaBatchScheduler,Batch

class BatchSchedulingTests(unittest.TestCase):
 def ready(self,s,rid,now=0):
  entry=s.reserve(rid,object());s.mark_admitted(rid);s.queue_decode(rid,32,now);return entry
 def test_full_batch_and_other_replica_admission(self):
  s=ReplicaBatchScheduler([2,3],width=2)
  self.ready(s,'a');self.ready(s,'b');batch=s.dispatchable(0)[0]
  self.assertEqual(batch.replica,2)
  self.assertEqual(self.ready(s,'c').replica,3)
  self.assertEqual([e.request_id for e in s.finish(batch,True)],['a','b'])
  self.assertEqual(s.requests['c'].state,'queued')
 def test_partial_batch_wait_is_bounded(self):
  s=ReplicaBatchScheduler([2],max_wait_seconds=.02);self.ready(s,'a',1)
  self.assertEqual(s.dispatchable(1.01),[])
  batch=s.dispatchable(1.03)[0];self.assertEqual(batch.requests,((0,'a'),))
  self.assertIsNone(s.reserve('b',object()))
  s.finish(batch,True);self.assertIsNotNone(s.reserve('b',object()))
 def test_no_dispatch_over_admitting_or_ready_slot(self):
  s=ReplicaBatchScheduler([2],width=2,max_wait_seconds=0);self.ready(s,'a')
  s.reserve('b',object());self.assertEqual(s.dispatchable(100),[])
  s.mark_admitted('b');self.assertEqual(s.dispatchable(100),[])
  s.discard('b');self.assertEqual(s.dispatchable(100)[0].requests,((0,'a'),))
 def test_failure_and_stale_completion_never_recycle(self):
  s=ReplicaBatchScheduler([2],width=1);self.ready(s,'a');batch=s.dispatchable(0)[0]
  with self.assertRaises(RuntimeError):s.finish(Batch(2,batch.generation,batch.requests),True)
  with self.assertRaises(RuntimeError):s.finish(batch,False)
  self.assertIn('a',s.requests);self.assertIsNone(s.reserve('b',object()))
  with self.assertRaises(RuntimeError):s.discard('a')
 def test_fence_and_duplicate_guards(self):
  s=ReplicaBatchScheduler([2]);s.reserve('a',object())
  with self.assertRaises(RuntimeError):s.queue_decode('a',32,0)
  with self.assertRaises(ValueError):s.reserve('a',object())
  s.mark_admitted('a');s.queue_decode('a',32,0)
  with self.assertRaises(RuntimeError):s.discard('a')
 def test_wrong_thread_is_rejected(self):
  s=ReplicaBatchScheduler([2]);errors=[]
  def call():
   try:s.reserve('a',object())
   except RuntimeError as e:errors.append(str(e))
  t=threading.Thread(target=call);t.start();t.join();self.assertEqual(len(errors),1)
if __name__=='__main__':unittest.main()
