import unittest,time
from hetero_p2p.llmd import StageBroker,StageError

class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.b=StageBroker('koni',lambda rid,prompt:self.calls.append(('prepare',rid)) or rid,
            lambda handle,prompt,count:self.calls.append(('decode',handle,count)) or {'choices':[{'text':'ok'}]},
            lambda handle:self.calls.append(('discard',handle)),capacity=1,ttl=10)
        self.body=dict(model='koni',prompt=[1,2],max_tokens=1,temperature=0,stream=False)
    def test_shared_storage_rewrite(self):
        self.b.execute('prefill','id',self.body)
        response=self.b.execute('decode','id',dict(self.body,max_tokens=32,cache_hit_threshold=0))
        self.assertEqual(response['choices'][0]['text'],'ok');self.assertIn(('decode','id',32),self.calls)
    def test_missing_prefill(self):
        with self.assertRaises(StageError):self.b.execute('decode','id',self.body)
    def test_fingerprint(self):
        self.b.execute('prefill','id',self.body)
        with self.assertRaises(StageError):self.b.execute('decode','id',dict(self.body,prompt=[3,4]))
    def test_idempotent_prefill(self):
        self.b.execute('prefill','id',self.body);self.b.execute('prefill','id',self.body)
        self.assertEqual(self.calls,[('prepare','id')])
    def test_capacity(self):
        self.b.execute('prefill','id',self.body)
        with self.assertRaises(StageError) as cm:self.b.execute('prefill','other',self.body)
        self.assertEqual(cm.exception.status,429)
    def test_expire_ready(self):
        self.b.execute('prefill','id',self.body);self.b.entries['id'].created-=11
        self.b.execute('prefill','other',self.body);self.assertIn(('discard','id'),self.calls)
    def test_failed_decode_quarantines(self):
        self.b.execute('prefill','id',self.body)
        def fail(*args):raise RuntimeError('device error')
        self.b.decode=fail
        with self.assertRaises(RuntimeError):self.b.execute('decode','id',self.body)
        self.b.entries['id'].created-=20;self.b._expire();self.assertEqual(self.b.entries['id'].state,'poisoned')
    def test_unsupported_sampling_and_stream(self):
        for patch in [dict(temperature=1),dict(stream=True),dict(top_p=.9),dict(prompt='text')]:
            with self.assertRaises(StageError):self.b.execute('prefill','id',dict(self.body,**patch))

if __name__=='__main__':unittest.main()
