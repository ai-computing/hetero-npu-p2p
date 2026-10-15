import unittest
try:
 import torch
except ModuleNotFoundError:
 torch=None
if torch is not None:
 from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker,PackKVPayload,PackPrefixKVPayload
@unittest.skipIf(torch is None,"optional torch adapter dependency is not installed")
class BoundedPackerTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(4);torch.manual_seed(123)
  self.p=BoundedPrefixPacker.__new__(BoundedPrefixPacker)
  self.p.device='cpu';self.p.layers=2;self.p.heads=2;self.p.head_dim=128;self.p.module=PackKVPayload()
  self.k=torch.randn(2,2,128,512).bfloat16();self.v=torch.randn(2,2,512,128).bfloat16()
 def test_chunk_boundaries_and_maximum(self):
  for length in [1,16,17,139,270,512]:
   with self.subTest(length=length):
    actual,bound=self.p.encode(self.k,self.v,length);expected,eb=PackPrefixKVPayload(length)(self.k,self.v)
    self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
    self.assertEqual(bound,float(eb))
 def test_reject_invalid_length(self):
  for value in [0,513,True,16.0]:
   with self.assertRaises(ValueError):self.p.encode(self.k,self.v,value)
 def test_reject_nonfinite(self):
  self.k[0,0,0,0]=float('nan')
  with self.assertRaises(ValueError):self.p.encode(self.k,self.v,17)
if __name__=='__main__':unittest.main()
