import unittest
from hetero_p2p.serving import BatchOptions,CompletionLedger,RngdBinding
class ServingTests(unittest.TestCase):
 def message(self,ids,kind='request_complete',batch='g1'):
  return dict(batch_id=batch,kind=kind,passed=True,requests=[dict(request_id=i,tokens=[1,2],passed=True) for i in ids])
 def test_early_reply_retains_replica_until_final(self):
  s=CompletionLedger();s.start(2,'g1',{'a':2,'b':2})
  fresh,final=s.accept(2,self.message(['a']));self.assertFalse(final);self.assertEqual(len(fresh),1)
  with self.assertRaises(RuntimeError):s.start(2,'g2',{'c':2})
  fresh,final=s.accept(2,self.message(['a','b'],'batch_complete'));self.assertTrue(final);self.assertEqual([r['request_id'] for r in fresh],['b'])
  s.start(2,'g2',{'c':2})
  with self.assertRaises(RuntimeError):s.accept(2,self.message(['a','b'],'batch_complete'))
 def test_invalid_messages_retain_lease(self):
  for msg in [self.message(['a','a']),self.message(['x']),self.message(['a'],'batch_complete'),self.message(['a'],batch='old')]:
   s=CompletionLedger();s.start(2,'g1',{'a':2,'b':2})
   with self.assertRaises(RuntimeError):s.accept(2,msg)
   self.assertIn(2,s.active)
 def test_changed_final_rejected(self):
  s=CompletionLedger();s.start(2,'g1',{'a':2});s.accept(2,self.message(['a']))
  msg=self.message(['a'],'batch_complete');msg['requests'][0]['tokens']=[3,4]
  with self.assertRaises(RuntimeError):s.accept(2,msg)
  self.assertIn(2,s.active)
 def test_options(self):
  self.assertEqual(BatchOptions(8,16).width,8)
  for args in [(8,4),(0,8),(4,8,float('nan'))]:
   with self.assertRaises(ValueError):BatchOptions(*args)
 def test_device_binding(self):
  b=RngdBinding.from_path('/dev/rngd/npu2ch0');self.assertEqual(b.index,2);self.assertEqual(b.torch_device,'furiosa:2')
  for path in ['/dev/rngd/npu02ch0','/dev/rngd/npu2ch1','/dev/rngd/npu-1ch0']:
   with self.assertRaises(ValueError):RngdBinding.from_path(path)
 def test_sparse_mapping_must_match_open_compute_channels(self):
  b=RngdBinding.from_path('/dev/rngd/npu3ch0',2);self.assertEqual(b.torch_device,'furiosa:2')
  b.verify_open_channels(['/dev/rngd/npu3ch0','/dev/rngd/npu3ch1r'])
  for paths in [[],['/dev/rngd/npu2ch0'],['/dev/rngd/npu3ch0','/dev/rngd/npu1ch0']]:
   with self.assertRaises(RuntimeError):b.verify_open_channels(paths)
if __name__=='__main__'  :unittest.main()
