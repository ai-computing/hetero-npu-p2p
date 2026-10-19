import contextlib,io,os,runpy,unittest
from unittest import mock
try:
    from rm_core import KVLayout,MemoryTier,NoHardware,TransportError,registry
    from rm_core.conformance import check_kv_transport
    from hetero_p2p.airm import P2PBinding,P2PTransport
    from hetero_p2p.core import SourceAllocation,DestinationAllocation
    from hetero_p2p.llmd import StageBroker,load_kv_transport
except ImportError:  # rm-core is a declared dependency; tolerate checkouts that lack it.
    P2PTransport=None

FORMAT='dlfloat16-token-major-v1'

class Backend:
    def __init__(self):self.plans=[]
    def transfer(self,p):self.plans.append(p);return dict(cqe_result=0,completed=p.commands,requested=p.commands,bytes_requested=p.bytes)

@unittest.skipIf(P2PTransport is None,'rm-core not installed')
class P2PTransportTests(unittest.TestCase):
    def setUp(self):
        # 1 layer x 1 KV head x 128 dims, context 32: one 4 KiB segment per cache for 16 tokens.
        self.fd=os.memfd_create('cpu-validation-only');os.ftruncate(self.fd,16384)
        self.src=MemoryTier('rngd1','device-hbm',48<<30,'host-a/furiosa:1')
        self.dst=MemoryTier('atom3','device-dram',16<<30,'host-a/rebellions:3')
        self.layout=KVLayout(num_layers=1,num_kv_heads=1,head_dim=128,dtype='dlfloat16')
        self.backend=Backend();self.metrics=[]
        self.t=P2PTransport(backend=self.backend,metric_sink=self.metrics.append)
    def tearDown(self):os.close(self.fd)
    def binding(self,size=8192):
        return P2PBinding(SourceAllocation(64,size,object(),FORMAT),(DestinationAllocation(self.fd,object(),FORMAT),),
            ({'index':0,'block_index':0,'offset':0},{'index':1,'block_index':0,'offset':8192}),context=32)
    def plan(self,rid='r',tokens=(0,16)):
        self.t.bind(rid,self.binding());return self.t.plan(rid,self.src,self.dst,self.layout,self.layout,tokens)
    def test_supports_only_same_host_rngd_to_atom(self):
        self.assertTrue(self.t.supports(self.src,self.dst))
        self.assertFalse(self.t.supports(self.dst,self.src))
        self.assertFalse(self.t.supports(self.src,MemoryTier('atom','device-dram',1,'host-b/rebellions:3')))
        self.assertFalse(self.t.supports(self.src,MemoryTier('host','host-dram',1)))
    def test_plan_wraps_layout(self):
        p=self.plan()
        self.assertEqual(p.nbytes,16*self.layout.bytes_per_token());self.assertEqual(p.detail['spans'],'2')
    def test_plan_rejects(self):
        for tokens in [(16,16),(-1,4),(16,32)]:
            with self.assertRaises(TransportError):self.plan(tokens=tokens)
        with self.assertRaisesRegex(TransportError,'bind'):self.t.plan('none',self.src,self.dst,self.layout,self.layout,(0,16))
        bf16=KVLayout(num_layers=1,num_kv_heads=1,head_dim=128,dtype='bf16')
        self.t.bind('x',self.binding())
        with self.assertRaisesRegex(TransportError,'dlfloat16'):self.t.plan('x',self.src,self.dst,bf16,bf16,(0,16))
        with self.assertRaisesRegex(TransportError,'4 KiB'):self.plan(tokens=(0,8))
        with self.assertRaises(TransportError):self.t.plan('r',self.dst,self.src,self.layout,self.layout,(0,16))
    def test_submit_wait_fence_and_metrics(self):
        p=self.plan();h=self.t.submit(p);r=self.t.wait(h,1.0)
        self.assertTrue(r.ok);self.assertEqual(r.nbytes,p.nbytes);self.assertEqual(len(self.backend.plans),1)
        self.assertEqual([m.name for m in self.metrics],['kv_transfer_ms','kv_transfer_gbps'])
        self.assertTrue(all(m.tags['kv_transport']=='p2p' for m in self.metrics))
        with self.assertRaises(TransportError):self.t.wait(h,1.0)
        with self.assertRaises(TransportError):self.t.submit(p)
    def test_backend_failure_is_not_retried(self):
        class Broken:
            calls=0
            def transfer(self,p):Broken.calls+=1;raise RuntimeError('PDMA failed (5)')
        t=P2PTransport(backend=Broken());t.bind('r',self.binding())
        r=t.wait(t.submit(t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))),1.0)
        self.assertFalse(r.ok);self.assertIn('PDMA failed',r.error);self.assertEqual(Broken.calls,1);self.assertEqual(list(t.measurements),[])
    def test_failing_metric_sink_keeps_fenced_result(self):
        def sink(row):raise ValueError('sink down')
        t=P2PTransport(backend=self.backend,metric_sink=sink);t.bind('r',self.binding())
        r=t.wait(t.submit(t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))),1.0)
        self.assertTrue(r.ok);self.assertEqual(len(t.measurements),2)
    def test_plan_errors_are_transport_errors(self):
        bad=P2PBinding(self.binding().source,self.binding().destinations,({'block_index':0,'offset':0},)*2,context=32)
        self.t.bind('r',bad)
        with self.assertRaises(TransportError):self.t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))
        dead=os.memfd_create('dead');b=self.binding();os.close(dead)
        self.t.bind('r',P2PBinding(b.source,(DestinationAllocation(dead,object(),FORMAT),),b.mappings,context=32))
        with self.assertRaises(TransportError):self.t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))
    def test_binding_is_one_shot(self):
        self.t.bind('r',self.binding())
        with self.assertRaises(TransportError):self.t.plan('r',self.src,self.dst,self.layout,self.layout,(3,16))
        with self.assertRaisesRegex(TransportError,'bind'):self.t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))
    def test_device_and_block_mismatch_rejected_at_plan(self):
        t=P2PTransport(device='/dev/rngd/npu3ch0',backend=self.backend);t.bind('r',self.binding())
        with self.assertRaisesRegex(TransportError,'npu3ch0'):t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))
        blocked=KVLayout(num_layers=1,num_kv_heads=1,head_dim=128,dtype='dlfloat16',block_tokens=64)
        self.t.bind('r',self.binding())
        with self.assertRaisesRegex(TransportError,'block_tokens'):self.t.plan('r',self.src,self.dst,blocked,blocked,(0,16))
    def test_backend_setup_failure_keeps_plan(self):
        t=P2PTransport(helper=None,backend=None);t.helper=None;t.bind('r',self.binding())
        p=t.plan('r',self.src,self.dst,self.layout,self.layout,(0,16))
        with self.assertRaisesRegex(TransportError,'HETERO_P2P_HELPER'):t.submit(p)
        t._backend=self.backend;self.assertTrue(t.wait(t.submit(p),1.0).ok)
    def test_no_hardware_matches_cli_main_module(self):
        # `python -m rm_core.conformance` re-executes that module; since rm-core 0.1.1 it shares
        # rm_core.errors.NoHardware, so the CLI reports SKIP instead of FAIL.
        cli=runpy.run_module('rm_core.conformance',run_name='rm_core_conformance_cli');out=io.StringIO()
        with mock.patch.object(registry,'load',return_value=self.t),contextlib.redirect_stdout(out),\
             mock.patch('hetero_p2p.airm.probe',return_value=dict(supported=False,checks={'device_path':False})):
            self.assertEqual(cli['main'](['airm.kv_transports','p2p']),0)
        self.assertIs(cli['NoHardware'],NoHardware)
        self.assertRegex(out.getvalue(),r'^SKIP  airm.kv_transports:p2p: .*device_path')
    @unittest.skipUnless('p2p' in (registry.available('airm.kv_transports') if P2PTransport else {}),'package not installed')
    def test_entry_point_resolves(self):
        self.assertIsInstance(load_kv_transport('p2p'),P2PTransport)
    def test_rm_core_conformance_with_fake_backend(self):
        self.t.bind('conf-1',self.binding())
        check_kv_transport(self.t,tiers=(self.src,self.dst),layout=self.layout)
    def test_conformance_case_skips_without_hardware(self):
        healthy=dict(supported=True,checks={})
        with mock.patch('hetero_p2p.airm.probe',return_value=dict(supported=False,checks={'atom_driver':False,'device_path':True})):
            with self.assertRaisesRegex(NoHardware,'atom_driver'):self.t.conformance_case()
        with mock.patch('hetero_p2p.airm.probe',return_value=healthy),mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(NoHardware,'HETERO_P2P_HELPER'):P2PTransport().conformance_case()
            with self.assertRaisesRegex(NoHardware,'FIXTURE'):P2PTransport(helper=__file__).conformance_case()
    def test_conformance_case_uses_fixture(self):
        fixture=lambda t:(self.src,self.dst,self.layout,self.binding())
        with mock.patch('hetero_p2p.airm.probe',return_value=dict(supported=True,checks={})),\
             mock.patch.dict(os.environ,{'HETERO_P2P_CONFORMANCE_FIXTURE':f'{__name__}:FIXTURE'}),\
             mock.patch(f'{__name__}.FIXTURE',fixture,create=True):
            t=P2PTransport(helper=__file__,backend=self.backend)
            check_kv_transport(t)
        self.assertEqual(len(self.backend.plans),1)
    def test_llmd_selects_transport_by_name(self):
        b=StageBroker('koni',None,None,None)
        self.assertEqual(b.kv_transport,'p2p')
        with self.assertRaises(ValueError):StageBroker('koni',None,None,None,kv_transport='')
        self.assertEqual(load_kv_transport('loopback').name,'loopback')

if __name__=='__main__':unittest.main()
