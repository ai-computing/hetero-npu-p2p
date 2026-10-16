"""Explain completed scenarios using producer-side monotonic event timestamps."""
import argparse,json,statistics
from pathlib import Path


def analyze(run):
    run=Path(run)
    audit=json.loads((run/'analysis.json').read_text())
    http=json.loads((run/'http-results.json').read_text())
    assert http['passed'] and audit['exact_logits'] and audit['ownership_violations']==0
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    ready={};first={};starts=[];ends={};copies={}
    for e in events:
        if e['stage']=='prefill_ready':
            assert e['request_id'] not in ready
            ready[e['request_id']]=e['time']
        elif e['stage']=='p2p_complete':copies[e['request_id']]=e['time']
        elif e['stage']=='iteration_start':
            starts.append(e)
            for q in e['requests']:
                if q['produced']==0:
                    assert q['id'] not in first
                    first[q['id']]=e['time']
        elif e['stage']=='iteration_complete':ends[(e['replica'],e['generation'])]=e
    rows=[]
    for row in http['rows']:
        if row['scenario'] in ('probe','warm'):continue
        ids={q['id'] for q in row['requests']}
        groups=[e for e in starts if any(q['id'] in ids for q in e['requests'])]
        assert groups and all(all(q['id'] in ids for q in e['requests']) for e in groups)
        for rid in ids:assert ready[rid]<=copies[rid]<=first[rid]
        steps=sum(e['steps'] for e in groups)
        worker=sum(ends[(e['replica'],e['generation'])]['seconds'] for e in groups)
        durations={}
        for n in sorted({q['output_tokens'] for q in row['requests']}):
            selected=[q['id'] for q in row['requests'] if q['output_tokens']==n]
            durations[str(n)]=dict(
                median_prefill_ready_to_first_dispatch_seconds=statistics.median(first[rid]-ready[rid] for rid in selected),
                median_copy_complete_to_first_dispatch_seconds=statistics.median(first[rid]-copies[rid] for rid in selected))
        rows.append(dict(scenario=row['scenario'],throughput=row['tokens_per_second'],
            steps_across_replicas=steps,worker_ms_per_step=1000*worker/steps,
            useful_lane_fraction=sum(e['steps']*len(e['requests']) for e in groups)/sum(e['steps']*e['compiled_batch'] for e in groups),
            waits_by_output=durations))
    return dict(scope='Producer event times: prefill-ready to first decode dispatch includes queue/control/admission. Copy-complete to dispatch includes collection. Neither is client TTFT nor device-only time.',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.run),indent=2))
