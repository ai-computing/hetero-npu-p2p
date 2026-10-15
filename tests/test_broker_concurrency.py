import concurrent.futures,threading,unittest
from hetero_p2p.llmd import StageBroker,StageError
BODY={'model':'test','prompt':[1,2], 'max_tokens':4}
class BrokerConcurrency(unittest.TestCase):
 def test_two_decodes_can_join_one_batch(self):
  barrier=threading.Barrier(2)
  def decode(handle,prompt,count):
   barrier.wait(timeout=2)
   return {'choices':[{'text':handle}]}
  broker=StageBroker('test',lambda rid,t:rid,decode,lambda h:None,capacity=2)
  for rid in ['a','b']:broker.execute('prefill',rid,BODY)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   futures=[pool.submit(broker.execute,'decode',rid,BODY) for rid in ['a','b']]
   for f in futures:self.assertIsInstance(f.result(timeout=3),dict)
  self.assertEqual(broker.entries,{})
 def test_pending_decode_does_not_block_other_admission(self):
  entered=threading.Event();release=threading.Event()
  def decode(*args):entered.set();self.assertTrue(release.wait(3));return {'choices':[]}
  broker=StageBroker('test',lambda rid,t:rid,decode,lambda h:None,capacity=2)
  broker.execute('prefill','a',BODY)
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   a=pool.submit(broker.execute,'decode','a',BODY)
   self.assertTrue(entered.wait(2))
   try:
    with self.assertRaises(StageError) as e:broker.execute('decode','a',BODY)
    self.assertEqual(e.exception.status,409)
    pool.submit(broker.execute,'prefill','b',BODY).result(timeout=1)
   finally:release.set()
   a.result(timeout=2)
  self.assertEqual(broker.entries['b'].state,'ready')
 def test_failed_expiry_does_not_strand_other_ready_entries(self):
  def discard(handle):raise RuntimeError('uncertain release')
  broker=StageBroker('test',lambda rid,t:rid,lambda *a:{'choices':[]},discard,capacity=3)
  for rid in ['a','b']:broker.execute('prefill',rid,BODY)
  for entry in broker.entries.values():entry.created-=100
  with self.assertRaises(RuntimeError):broker._expire()
  self.assertEqual(broker.entries['a'].state,'poisoned')
  self.assertEqual(broker.entries['b'].state,'ready')
if __name__=='__main__':unittest.main()
