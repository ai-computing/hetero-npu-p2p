"""ai-rm protocol v0.1 KVTransport adapter (entry point `airm.kv_transports:p2p`).

Wraps the existing completion-fenced path: `TokenMajorLayout.prefix_plan` builds the copy plan and
`RngdPdmaBackend.transfer` runs the native helper, which returns only after every CQE is reaped.
No transfer logic lives here and there is no host-staging fallback (PROTOCOL §4.5).

rm-core plans by tier and layout; live SDK allocations cannot be discovered from those, so the
owning process binds them per request with `bind()` before `plan()`.
"""
from dataclasses import dataclass
from collections import deque
import importlib,os,sys,time,uuid
from rm_core import TransferHandle,TransferPlan,TransferResult,TransportError
from rm_core.metrics import Measurement
from .backend import RngdPdmaBackend,probe
from .core import DestinationAllocation,SourceAllocation
from .layout import TokenMajorLayout

DEVICE_KINDS=('device-hbm','device-dram')


def _no_hardware():
    # rm-core v0.1.0: `python -m rm_core.conformance` runs that module as __main__, so its SKIP
    # handler catches __main__.NoHardware, not the imported class. Raise one matching both.
    from rm_core.conformance import NoHardware
    main=sys.modules.get('__main__')
    other=getattr(main,'NoHardware',None)
    if getattr(getattr(main,'__spec__',None),'name',None)=='rm_core.conformance' and other not in (None,NoHardware):
        return type('NoHardware',(NoHardware,other),{})
    return NoHardware


@dataclass(frozen=True)
class P2PBinding:
    """Live allocations for one request: packed RNGD prefix and exported ATOM caches.

    `mappings` are the ATOM cache records from `atom_experimental.ExportedCaches.mappings`.
    The caller keeps every owner alive until decode finishes; `close()` never releases them.
    """
    source: SourceAllocation
    destinations: tuple[DestinationAllocation, ...]
    mappings: tuple[dict, ...]
    context: int
    slot: int = 0
    batch: int = 1


def _vendor(tier):
    node,_,device=tier.accelerator_id.partition('/')
    return node,device.split(':',1)[0]


class P2PTransport:
    """RNGD device memory -> ATOM device memory on the same host (validated ABI scope only).

    Keyword arguments (all optional): `helper` path of `build/rngd-pdma` (default env
    HETERO_P2P_HELPER), `device` RNGD PE path (default env HETERO_P2P_DEVICE or
    /dev/rngd/npu1ch0), `backend` an object with `transfer(plan)` (tests), `metric_sink` a callable
    receiving each `rm_core.metrics.Measurement`. Every plan must be submitted and every handle
    waited (or the transport closed); unsubmitted plans and unwaited results are kept until then.
    """
    name='p2p'

    def __init__(self,helper=None,device=None,backend=None,metric_sink=None):
        self.helper=helper or os.environ.get('HETERO_P2P_HELPER')
        self.device=device or os.environ.get('HETERO_P2P_DEVICE','/dev/rngd/npu1ch0')
        self._backend,self.metric_sink=backend,metric_sink
        self._bindings,self._plans,self._results={}, {}, {}
        self.measurements=deque(maxlen=1024)

    def supports(self,src,dst):
        if src.kind not in DEVICE_KINDS or dst.kind not in DEVICE_KINDS:return False
        (src_node,src_vendor),(dst_node,dst_vendor)=_vendor(src),_vendor(dst)
        return src_vendor=='furiosa' and dst_vendor=='rebellions' and src_node==dst_node!=''

    def bind(self,request_id,binding):
        if not isinstance(binding,P2PBinding):raise TransportError(f'{self.name}: bind() needs a P2PBinding')
        self._bindings[request_id]=binding

    def plan(self,request_id,src,dst,src_layout,dst_layout,tokens):
        # A binding is one-shot: any plan() for its request consumes it, so a rejected plan never
        # leaves stale allocations behind for a later call.
        binding=self._bindings.pop(request_id,None)
        if not self.supports(src,dst):
            raise TransportError(f'{self.name}: only same-host RNGD -> ATOM device memory is validated, not {src.name} -> {dst.name}')
        src.validate();dst.validate();src_layout.validate();dst_layout.validate()
        start,end=tokens
        if type(start) is not int or type(end) is not int or not 0<=start<end:
            raise TransportError(f'{self.name}: bad token range {tokens}')
        if start!=0:raise TransportError(f'{self.name}: only prefix transfers [0, end) are supported')
        if (src_layout!=dst_layout or src_layout.dtype!='dlfloat16' or src_layout.order!='token-major'
                or src_layout.block_tokens!=1):
            raise TransportError(f'{self.name}: needs identical token-major dlfloat16 layouts (block_tokens=1); convert in the model adapter')
        if binding is None:raise TransportError(f'{self.name}: no live allocation bound for {request_id!r}; call bind() first')
        if binding.source.device!=self.device:
            raise TransportError(f'{self.name}: source lives on {binding.source.device}, transport drives {self.device}')
        layout=TokenMajorLayout(2*src_layout.num_layers,src_layout.num_kv_heads,binding.context,src_layout.head_dim)
        try:
            inner=layout.prefix_plan(binding.source,binding.destinations,list(binding.mappings),end,binding.slot,binding.batch)
        except (ValueError,KeyError,TypeError,OSError) as e:
            raise TransportError(f'{self.name}: {type(e).__name__}: {e}') from e
        nbytes=end*src_layout.bytes_per_token()  # prefix_plan already pins source.size to this
        plan_id=uuid.uuid4().hex;self._plans[plan_id]=inner
        return TransferPlan(request_id,src,dst,src_layout,dst_layout,(start,end),nbytes,
                            {'plan_id':plan_id,'commands':str(inner.commands),'spans':str(len(inner.spans)),'device':binding.source.device})

    def _get_backend(self):
        if self._backend is None:
            if not self.helper:raise TransportError(f'{self.name}: set HETERO_P2P_HELPER to the built rngd-pdma helper')
            try:self._backend=RngdPdmaBackend(self.helper,self.device)
            except (OSError,ValueError,KeyError,RuntimeError) as e:raise TransportError(f'{self.name}: backend unavailable: {e}') from e
        return self._backend

    def submit(self,plan):
        """Run on the owning process's main thread (RngdPdmaBackend refuses other threads)."""
        plan_id=plan.detail.get('plan_id','')
        if plan_id not in self._plans:raise TransportError(f'{self.name}: unknown or already-submitted plan for {plan.request_id!r}')
        backend=self._get_backend()  # before consuming the plan, so a fixed environment can resubmit
        inner=self._plans.pop(plan_id);started=time.perf_counter()
        try:
            # Synchronous: the helper returns only after every CQE completed (completion fence).
            receipt=backend.transfer(inner);seconds=time.perf_counter()-started
            result=TransferResult(plan.request_id,True,receipt['bytes_requested'],seconds)
        except Exception as e:
            # The destination is now uncertain; the caller must poison its lease, not retry another path.
            result=TransferResult(plan.request_id,False,0,time.perf_counter()-started,f'{type(e).__name__}: {e}')
        if result.ok:self._report(result)
        token=uuid.uuid4().hex;self._results[token]=result
        return TransferHandle(plan.request_id,self.name,token)

    def wait(self,handle,timeout_s):
        result=self._results.pop(handle.token,None)
        if result is None:raise TransportError(f'{self.name}: unknown or already-waited handle {handle.token}')
        return result

    def _report(self,result):
        tags={'kv_transport':self.name,'component':'hetero-npu-p2p'}
        rows=[Measurement('kv_transfer_ms',result.seconds*1e3,tags)]
        if result.seconds>0:rows.append(Measurement('kv_transfer_gbps',result.nbytes/result.seconds/1e9,tags))
        for row in rows:
            self.measurements.append(row.validate())
            if self.metric_sink:
                try:self.metric_sink(row)
                except Exception:pass  # A metrics outage must not turn a fenced transfer into a failure.

    def close(self):
        """Drop pending bindings, plans and results. Allocation owners stay with the caller."""
        self._bindings.clear();self._plans.clear();self._results.clear()

    def conformance_case(self):
        """Real tiers on this host, or NoHardware.

        Live RNGD/ATOM allocations come from a caller-provided fixture named by
        HETERO_P2P_CONFORMANCE_FIXTURE=`module:callable`; it receives this transport and returns
        `(src_tier, dst_tier, layout, binding)` with owners kept alive by the fixture.
        """
        NoHardware=_no_hardware()
        health=probe(self.device)
        if not health['supported']:
            failed=sorted(k for k,v in health['checks'].items() if not v)
            raise NoHardware(f'{self.name}: unvalidated environment for {self.device} (failed: {", ".join(failed)})')
        if not self.helper or not os.path.exists(self.helper):
            raise NoHardware(f'{self.name}: native helper not found; set HETERO_P2P_HELPER')
        fixture=os.environ.get('HETERO_P2P_CONFORMANCE_FIXTURE')
        if not fixture:
            raise NoHardware(f'{self.name}: no live RNGD/ATOM allocation fixture; set HETERO_P2P_CONFORMANCE_FIXTURE')
        module,_,attr=fixture.partition(':')
        src,dst,layout,binding=getattr(importlib.import_module(module),attr)(self)
        self.bind('conf-1',binding)
        return src,dst,layout
