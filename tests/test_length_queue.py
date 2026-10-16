import unittest
from hetero_p2p.length_queue import LengthQueue
class LengthQueueTests(unittest.TestCase):
 def test_length_locality_and_oldest_fairness(self):
  q=LengthQueue()
  for i,n in enumerate([128,32,64,96,32,64,96,128]):q.add(str(i),n,i)
  batch=q.take(10,policy='length')
  self.assertEqual([j.output_tokens for j in batch],[128,128,96,96])
  self.assertEqual([j.request_id for j in q.take(11,policy='length')],['1','4','2','5'])
 def test_fifo_control(self):
  q=LengthQueue()
  for i in range(8):q.add(str(i),32,i)
  self.assertEqual([j.request_id for j in q.take(10,policy='fifo')],['0','1','2','3'])
 def test_partial_wait_and_prefill_collection(self):
  q=LengthQueue();q.add('a',32,0)
  self.assertEqual(q.take(.01),[]);self.assertEqual(q.take(1,more_prefills=True),[])
  self.assertEqual(q.take(1)[0].request_id,'a')
 def test_duplicates_and_limits(self):
  q=LengthQueue();q.add('a',32,0)
  with self.assertRaises(ValueError):q.add('a',64,1)
  with self.assertRaises(ValueError):q.take(1,policy='invalid')
if __name__=='__main__':unittest.main()
