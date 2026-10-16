"""Synthetic reporting fixtures, never hardware performance evidence."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('batch_abba',Path(__file__).parents[1]/'examples/koni/analyze_batch_abba.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class BatchAbbaTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.paths=[]
  for i,(batch,tps) in enumerate(zip((12,14,14,12),(100,120,118,102))):
   p=Path(self.tmp.name)/str(i);p.mkdir();self.paths.append(p)
   cfg=dict(batch_size=batch,context=512 if batch==12 else 448,atom_artifact=f'b{batch}',queue_capacity=96)
   req=[dict(output_tokens=128) for _ in range(84)]
   rows=[dict(scenario='probe',requests=[]),dict(scenario='warm',requests=req)]+[dict(scenario=f'sealed-fixed128-r{j}',requests=req,tokens_per_second=tps+j-1) for j in range(3)]
   audit=dict(exact_logits=True,ownership_violations=0,requests=368,cases=dict(fixed128=dict(sealed=dict(useful_lane_fraction=1.0,median_tps=tps))))
   for name,value in [('config',cfg),('http-results',dict(passed=True,rows=rows)),('analysis',audit),('result',dict(passed=True))]:
    (p/(name+'.json')).write_text(json.dumps(value))
   events=[dict(time=i*1000+j,stage='p2p_complete',requested=64,completed=64,cqe_result=0) for j in range(368)]
   (p/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
 def edit(self,index,name,fn):
  p=self.paths[index]/(name+'.json');v=json.loads(p.read_text());fn(v);p.write_text(json.dumps(v))
 def test_process_blocks_and_pair_direction(self):
  r=mod.analyze(self.paths)
  self.assertEqual(r['requests'],1472);self.assertEqual(r['process_median_by_batch'],{'12':101,'14':119})
  self.assertEqual(r['paired_b14_over_b12'],[1.2,118/102])
 def test_wrong_request_count_rejected(self):
  self.edit(2,'http-results',lambda v:v['rows'][2]['requests'].pop())
  with self.assertRaises(AssertionError):mod.analyze(self.paths)
 def test_failed_terminal_rejected(self):
  self.edit(3,'result',lambda v:v.update(passed=False))
  with self.assertRaises(AssertionError):mod.analyze(self.paths)
 def test_queue_confound_rejected(self):
  self.edit(2,'config',lambda v:v.update(queue_capacity=128))
  with self.assertRaisesRegex(AssertionError,'queue_capacity'):mod.analyze(self.paths)
 def test_duplicate_process_rejected(self):
  with self.assertRaises(ValueError):mod.analyze([self.paths[0]]*4)
