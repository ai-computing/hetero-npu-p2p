"""RNGD PDMA backend. No reverse import, raw-address discovery or silent fallback."""
from pathlib import Path
import hashlib,json,os,platform,re,signal,subprocess,tempfile,threading
from .core import TransferPlan

THUNK_SHA256 = '5cc0575543d1f10babc6a16c8fd1d54b373132c1f0f49dd53dd291809d298b09'

def probe(device='/dev/rngd/npu1ch0'):
    checks = {'linux_x86_64':platform.system()=='Linux' and platform.machine()=='x86_64',
              'device_path':bool(re.fullmatch(r'/dev/rngd/npu\d+ch0',device)) and Path(device).exists()}
    thunk=Path('/usr/lib/librbln-thunk.so.3.0.0')
    checks['atom_thunk_abi']=thunk.exists() and hashlib.sha256(thunk.read_bytes()).hexdigest()==THUNK_SHA256
    versions={}
    try:
        r=subprocess.run(['furiosa-smi','version'],capture_output=True,text=True,timeout=10)
        match=re.search(r'- device driver\s+([0-9.]+)',r.stdout)
        versions['furiosa_rngd']=match.group(1) if r.returncode==0 and match else None
    except (OSError,subprocess.TimeoutExpired):
        versions['furiosa_rngd']=None
    f=Path('/sys/module/rebellions/version')
    versions['rbln']=f.read_text().strip() if f.exists() else None
    checks['rngd_driver']=versions['furiosa_rngd']=='2026.4.0'
    checks['atom_driver']=versions['rbln']=='3.0.0'
    return dict(supported=all(checks.values()),checks=checks,versions=versions,device=device,
                validation_scope='RNGD1 PE0 -> ATOM2/3, caller-owned exportable DLFloat16 KV; same host',
                portability='Other same-version paths are configurable but require hardware validation; other versions fail closed')


class RngdPdmaBackend:
    def __init__(self, helper, device='/dev/rngd/npu1ch0'):
        self.helper=Path(helper).resolve();self.device=device
        manifest=json.loads(self.helper.with_suffix('.json').read_text())
        if hashlib.sha256(self.helper.read_bytes()).hexdigest()!=manifest['sha256']:
            raise RuntimeError('native helper does not match build manifest')
        if manifest.get('rngd_driver')!='2026.4.0':raise RuntimeError('unsupported helper build ABI')
        health=probe(device)
        if not health['supported']:raise RuntimeError(f'unsupported environment: {health}')

    def transfer(self, plan: TransferPlan):
        plan.validate()
        if plan.source.device != self.device:
            raise ValueError('source allocation and DMA device do not match')
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Use an owning worker process: hardware transfers must run on its main thread')
        # A temporary private file carries only fd/offset metadata, never KV bytes.
        with tempfile.TemporaryDirectory(prefix='hetero-p2p-') as directory:
            path=Path(directory)/'plan.txt'
            path.write_text(''.join(f'{plan.destinations[s.destination].fd} {s.source_offset} {s.destination_offset} {s.length}\n' for s in plan.spans))
            fds=tuple(sorted({d.fd for d in plan.destinations}))
            # Delay process interruption until the synchronous native helper has reaped every CQE.
            previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
            try:
                proc=subprocess.run([str(self.helper),self.device,hex(plan.source.address),str(plan.source.size),str(path),str(plan.descriptors_per_command)],pass_fds=fds,capture_output=True,text=True)
                if proc.returncode:raise RuntimeError(f'PDMA failed ({proc.returncode}): {proc.stdout} {proc.stderr}')
                receipt=json.loads(proc.stdout)
                if (receipt.get('cqe_result')!=0 or receipt.get('completed')!=plan.commands
                    or receipt.get('requested')!=plan.commands or receipt.get('bytes_requested')!=plan.bytes):
                    raise RuntimeError(f'invalid completion receipt: {receipt}')
                return receipt
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK,previous)
