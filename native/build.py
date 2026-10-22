"""Build the native RNGD helpers against the installed driver headers, each with a sha256 build manifest.

rngd-pdma          build/rngd-pdma           RNGD -> ATOM KV (TransferPlan, RngdPdmaBackend); needs --liburing-root
librngd_dmabuf.so  build/librngd_dmabuf.so   RNGD <-> any imported DMA-BUF, e.g. CUDA memory (RngdDmabufBackend);
                                             raw io_uring syscalls, no liburing
"""
from pathlib import Path
import argparse,subprocess,json,hashlib
p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--driver-root',type=Path,default=Path('/usr/src/furiosa-driver-rngd-2026.4.0'))
p.add_argument('--liburing-root',type=Path,help='built liburing tree (rngd-pdma only)')
p.add_argument('--target',choices=('rngd-pdma','rngd-dmabuf','all'),default='all')
a=p.parse_args()
if a.target in ('rngd-pdma','all') and not a.liburing_root:p.error('--liburing-root is required to build rngd-pdma')
root=Path(__file__).resolve().parent.parent;out=root/'build';out.mkdir(exist_ok=True)
inc=['-I'+str(a.driver_root/'include'),'-I'+str(a.driver_root/'src')]
targets=[]
if a.target in ('rngd-pdma','all'):
    targets.append((out/'rngd-pdma','rngd_pdma.c',['-I'+str(a.liburing_root/'src/include')],[str(a.liburing_root/'src/liburing.a')]))
if a.target in ('rngd-dmabuf','all'):
    targets.append((out/'librngd_dmabuf.so','rngd_dmabuf.c',['-shared','-fPIC'],[]))
for binary,source,flags,libs in targets:
    cmd=['gcc','-Wall','-Wextra','-Werror','-O2',*flags,*inc,str(root/'native'/source),*libs,'-o',str(binary)]
    subprocess.run(cmd,check=True)
    binary.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((root/'native'/source).read_bytes()).hexdigest(),rngd_driver='2026.4.0',command=cmd),indent=2)+'\n')
    print(binary)
