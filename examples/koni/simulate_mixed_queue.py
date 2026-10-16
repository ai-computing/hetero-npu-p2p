"""Ideal all-ready scheduling diagnostic, not a hardware throughput prediction."""
import argparse
import json
from pathlib import Path
from hetero_p2p.length_queue import LengthQueue


def simulate(lengths, width, lookahead, policy, refill):
    queue = LengthQueue(width, lookahead, 0)
    for index, length in enumerate(lengths):
        queue.add(str(index), length, index)
    active = []
    steps = 0
    while queue.jobs or active:
        if not active or refill:
            selected = queue.take(len(lengths) + 1, policy=policy, limit=width-len(active))
            active.extend(job.output_tokens for job in selected)
        advance = min(active)
        steps += advance
        active = [remaining-advance for remaining in active if remaining>advance]
    return dict(steps=steps, useful_lane_fraction=sum(lengths)/(width*steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    http = json.loads((args.run/'http-results.json').read_text())
    assert http['passed'], 'use a completed run'
    events = [json.loads(line) for line in (args.run/'events.jsonl').read_text().splitlines()]
    width = json.loads((args.run/'config.json').read_text())['batch_size']
    rows = []
    for row in http['rows']:
        if '-mixed-' not in row['scenario']:
            continue
        counts = {q['id']: q['output_tokens'] for q in row['requests']}
        order = [e['request_id'] for e in events if e['stage']=='decode_start' and e['request_id'] in counts]
        assert len(order)==len(counts)==len(set(order))
        for lookahead in (32, len(counts)):
            for policy in ('length', 'padding'):
                for refill in (False, True):
                    rows.append(dict(scenario=row['scenario'], width=width, lookahead=lookahead,
                                     policy=policy, refill=refill,
                                     **simulate([counts[rid] for rid in order], width, lookahead, policy, refill)))
    args.output.write_text(json.dumps(dict(
        scope='All requests available at time zero; preserves observed decode arrival order only. Ignores prefill, DMA, HTTP, step-time variation. Diagnostic, not predicted tokens/s.',
        rows=rows), indent=2))


if __name__ == '__main__':
    main()
