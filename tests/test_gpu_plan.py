import os,subprocess,unittest
from pathlib import Path
from hetero_p2p.gpu import CudaRegion,DmabufPlan,DmabufSpan,plan_copy,SPAN

MiB=1024**2

class MemRegion(CudaRegion):
    """CPU stand-in: memfd slices instead of CUDA DMA-BUF exports"""
    def fd(self,index):
        if index not in self._fds:
            self._fds[index]=os.memfd_create('slice');os.ftruncate(self._fds[index],min(self.slice_bytes,self.size-index*self.slice_bytes))
        return self._fds[index]

class PlanTests(unittest.TestCase):
    def setUp(self):self.r=MemRegion(1<<30,40*MiB,object(),slice_bytes=16*MiB)
    def tearDown(self):self.r.close()
    def test_locate_splits_at_slices(self):
        self.assertEqual(self.r.locate(15*MiB,3*MiB),[(0,15*MiB,MiB),(1,0,2*MiB)])
        with self.assertRaises(ValueError):self.r.locate(39*MiB,2*MiB)
    def test_plan_copy_spans_and_commands(self):
        p=plan_copy('write',self.r,0,1<<20,40*MiB,descriptors_per_command=4).validate()
        self.assertEqual(p.bytes,40*MiB)
        self.assertEqual(len(p.spans),20)                       # 2 MiB spans
        self.assertEqual(p.commands,2+2+1)                      # 8, 8, 4 spans per 16/16/8 MiB slice, 4 per command
        self.assertEqual([s.npu_offset for s in p.spans[:2]],[1<<20,(1<<20)+SPAN])
        self.assertEqual(len({s.fd for s in p.spans}),3)
    def test_alignment_bounds_and_overlap(self):
        fd=self.r.fd(0)
        bad=[DmabufSpan(fd,1,0,4096)],[DmabufSpan(fd,0,0,3*MiB)],[DmabufSpan(fd,0,16*MiB,4096)]
        for spans in bad:
            with self.assertRaises(ValueError):DmabufPlan('read',tuple(spans),object()).validate()
        with self.assertRaisesRegex(ValueError,'overlapping'):
            DmabufPlan('write',(DmabufSpan(fd,0,0,8192),DmabufSpan(fd,4096,8192,4096)),object()).validate()
        alias=os.dup(fd)
        try:
            with self.assertRaisesRegex(ValueError,'overlapping'):   # FD aliases of one DMA-BUF
                DmabufPlan('read',(DmabufSpan(fd,0,0,8192),DmabufSpan(alias,8192,4096,4096)),object()).validate()
        finally:os.close(alias)
        DmabufPlan('read',(DmabufSpan(fd,0,0,8192),DmabufSpan(fd,0,8192,4096)),object()).validate()  # reads may share a source
    def test_region_requires_alignment(self):
        with self.assertRaises(ValueError):CudaRegion(4097,4096,object())
        with self.assertRaises(ValueError):CudaRegion(4096,4096,object(),slice_bytes=512*MiB)

class HelperRejectionTests(unittest.TestCase):
    helper=Path(__file__).resolve().parent.parent/'build'/'rngd-dmabuf'
    @unittest.skipUnless(helper.exists(),'build/rngd-dmabuf not built (native/build.py)')
    def test_invalid_plans_rejected_before_any_device_access(self):
        fd=os.memfd_create('x');os.ftruncate(fd,8192)
        try:
            for rows,dev in ((f'{fd} 1 0 4096\n','/dev/rngd/npu0ch0'),(f'{fd} 0 0 16384\n','/dev/rngd/npu0ch0'),
                             (f'{fd} 0 0 4096\n','/dev/rngd/npu0ch1'),(f'{fd} 0 0 4096\n{fd} 0 4096 4096\n','/dev/rngd/npu0ch0')):
                plan=Path(os.environ.get('TMPDIR','/tmp'))/f'plan-{os.getpid()}.txt';plan.write_text(rows)
                direction='write'
                r=subprocess.run([str(self.helper),dev,direction,str(plan),'8','2','1'],pass_fds=(fd,),capture_output=True,text=True)
                plan.unlink()
                self.assertEqual(r.returncode,2,(rows,r.stderr))
        finally:os.close(fd)
