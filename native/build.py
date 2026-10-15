from pathlib import Path
import argparse,subprocess,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--driver-root',type=Path,default=Path('/usr/src/furiosa-driver-rngd-2026.4.0'));p.add_argument('--liburing-root',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent.parent;out=root/'build';out.mkdir(exist_ok=True);binary=out/'rngd-pdma'
cmd=['gcc','-Wall','-Wextra','-Werror','-O2','-I'+str(a.driver_root/'include'),'-I'+str(a.driver_root/'src'),'-I'+str(a.liburing_root/'src/include'),str(root/'native/rngd_pdma.c'),str(a.liburing_root/'src/liburing.a'),'-o',str(binary)]
subprocess.run(cmd,check=True)
binary.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((root/'native/rngd_pdma.c').read_bytes()).hexdigest(),rngd_driver='2026.4.0',command=cmd),indent=2)+'\n')
print(binary)
