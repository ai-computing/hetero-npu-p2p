import os,unittest
from hetero_p2p.core import SourceAllocation,DestinationAllocation,CopySpan,TransferPlan,CopySession
from hetero_p2p.layout import TokenMajorLayout

class Backend:
    def transfer(self,p):return dict(cqe_result=0,completed=p.commands,requested=p.commands,bytes_requested=p.bytes)

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.fd=os.memfd_create('cpu-validation-only');os.ftruncate(self.fd,16384)
        self.source=SourceAllocation(64,8192,object(),'test')
        self.dest=DestinationAllocation(self.fd,object(),'test')
        self.plan=TransferPlan(self.source,(self.dest,),(CopySpan(0,0,0,4096),CopySpan(0,4096,4096,4096)),2)
    def tearDown(self):os.close(self.fd)
    def test_non_page_aligned_source_valid(self):self.plan.validate()
    def test_bounds(self):
        with self.assertRaises(ValueError):TransferPlan(self.source,(self.dest,),(CopySpan(0,8192,0,4096),),1).validate()
    def test_alignment(self):
        with self.assertRaises(ValueError):TransferPlan(self.source,(self.dest,),(CopySpan(0,0,1,4096),),1).validate()
    def test_format_mismatch(self):
        with self.assertRaises(ValueError):TransferPlan(self.source,(DestinationAllocation(self.fd,object(),'other'),),self.plan.spans,2).validate()
    def test_alias_overlap(self):
        duplicate=os.dup(self.fd)
        try:
            with self.assertRaisesRegex(ValueError,'overlapping'):
                TransferPlan(self.source,(self.dest,DestinationAllocation(duplicate,object(),'test')),(CopySpan(0,0,0,4096),CopySpan(1,4096,0,4096)),1).validate()
        finally:os.close(duplicate)
    def test_lease_and_late_completion(self):
        s=CopySession(Backend());seen=[];s.admit('a',self.plan,lambda:seen.append(1));s.start_decode('a')
        with self.assertRaises(RuntimeError):s.admit('b',self.plan,lambda:None)
        with self.assertRaises(RuntimeError):s.finish_decode('old',True)
        s.finish_decode('a',True);self.assertEqual(s.state,'idle');self.assertEqual(seen,[1])
        with self.assertRaises(RuntimeError):s.finish_decode('a',True)
    def test_incomplete_dma_pins_owners(self):
        class Partial:
            def transfer(self,p):return dict(cqe_result=0,completed=0,requested=p.commands,bytes_requested=p.bytes)
        s=CopySession(Partial())
        with self.assertRaises(RuntimeError):s.admit('a',self.plan,lambda:self.fail('notifier before completion'))
        self.assertEqual(s.state,'poisoned');self.assertIs(s._plan,self.plan)
    def test_notify_failure_poisons(self):
        s=CopySession(Backend())
        def fail():raise RuntimeError('notify')
        with self.assertRaises(RuntimeError):s.admit('a',self.plan,fail)
        self.assertEqual(s.state,'poisoned')
    def test_failed_decode_cannot_release(self):
        s=CopySession(Backend());s.admit('a',self.plan,lambda:None);s.start_decode('a')
        with self.assertRaises(RuntimeError):s.finish_decode('a',False)
        self.assertEqual(s.state,'poisoned')
    def test_ready_cancel_only(self):
        s=CopySession(Backend());s.admit('a',self.plan,lambda:None);s.discard_ready('a');self.assertEqual(s.state,'idle')
    def test_layout_maps_prefix(self):
        layout=TokenMajorLayout(1,2,32,128,format='test')
        p=layout.prefix_plan(self.source,(self.dest,),[dict(index=0,block_index=0,offset=0)],16)
        self.assertEqual([s.destination_offset for s in p.spans],[0,8192]);self.assertEqual(p.bytes,8192)

if __name__=='__main__':unittest.main()
