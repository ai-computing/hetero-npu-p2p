"""Rejection tests only: malformed plans must exit before opening any NPU."""
import os,subprocess,tempfile,json
from pathlib import Path
helper=Path(__file__).resolve().parent.parent/'build/rngd-pdma'
a=os.memfd_create('scatter-validation-a');b=os.memfd_create('scatter-validation-b')
for fd in [a,b]:os.ftruncate(fd,2097152)
cases={
 'unaligned_source':(f'{a} 1 0 4096\n',1),
 'unaligned_destination':(f'{a} 0 1 4096\n',1),
 'unaligned_length':(f'{a} 0 0 2048\n',1),
 'source_overflow':(f'{a} 2097152 0 4096\n',1),
 'destination_overflow':(f'{a} 0 2097152 4096\n',1),
 'overlapping_destination':(f'{a} 0 0 4096\n{a} 4096 0 4096\n',2),
 'mixed_fds_in_command':(f'{a} 0 0 4096\n{b} 4096 0 4096\n',2),
 'incomplete_group':(f'{a} 0 0 4096\n',8),
 'invalid_batch':(f'{a} 0 0 4096\n',9),
 'extra_field':(f'{a} 0 0 4096 garbage\n',1),
 'empty_plan':('',1),
}
results=[]
try:
 with tempfile.TemporaryDirectory() as directory:
  plan=Path(directory)/'plan.txt'
  for name,(text,batch) in cases.items():
   plan.write_text(text)
   p=subprocess.run([str(helper),'/dev/rngd/npu999999ch0','0x10000000','2097152',str(plan),str(batch)],pass_fds=(a,b),capture_output=True,text=True)
   assert p.returncode==2,(name,p.returncode,p.stdout,p.stderr)
   results.append({'case':name,'returncode':p.returncode,'passed':True})
finally:
 for fd in [a,b]:os.close(fd)
print(json.dumps(results,indent=2))
