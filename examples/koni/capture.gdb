set pagination off
set breakpoint pending on
set print thread-events off
set environment TORCH_RBLN_EAGER_MALLOC 1
unset environment RBLN_DUMMY_DEVICE
set environment TORCH_RBLN_DISABLE_FALLBACK compile_error,unsupported_op
python
import gdb,json,pathlib
import os
path=pathlib.Path(os.environ['HETERO_CAPTURE_DIR'])/'live-capture.json'
state={'contexts':[],'allocations':[]}
def save():
 state['pid']=gdb.selected_inferior().pid
 path.write_text(json.dumps(state))
class Return(gdb.FinishBreakpoint):
 def __init__(self,kind,data):
  super().__init__(gdb.newest_frame(),internal=True);self.kind=kind;self.data=data
 def stop(self):
  rax=int(gdb.parse_and_eval('$rax'))
  if self.kind=='context' and rax==0:
   ptr=int.from_bytes(bytes(gdb.selected_inferior().read_memory(self.data,8)),'little');state['contexts'].append(ptr)
  elif self.kind=='allocation' and rax:
   self.data['dva']=rax;state['allocations'].append(self.data)
  save();return False
class Entry(gdb.Breakpoint):
 def stop(self):
  reg=lambda r:int(gdb.parse_and_eval('$'+r))
  if self.location=='rblnCreateContext':Return('context',reg('rsi'))
  else:Return('allocation',{'context':reg('rdi'),'size':reg('rcx'),'flags':reg('r8')})
  return False
Entry('rblnCreateContext');Entry('rblnAllocateDeviceMemory')

class Disable(gdb.Breakpoint):
 def stop(self):
  if (path.parent/'capture-complete').exists():
   for bp in gdb.breakpoints():bp.enabled=False
   (path.parent/'breakpoints-disabled').write_text('All breakpoints disabled before warmups; GDB remains attached')
  return False
Disable('getppid')
end
run
