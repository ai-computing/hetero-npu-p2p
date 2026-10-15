"""Pre-DMA startup checks; never reset devices or terminate their owners."""
from pathlib import Path
import csv
import io
import json
import re
import socket
import subprocess
import time


def context_pids(output, devices=(0, 1, 2, 3)):
    if 'Context Information' not in output:
        raise RuntimeError('Unrecognized rbln-smi output; cannot establish idle state')
    section = output.split('Context Information', 1)[1]
    pids = []
    for line in section.splitlines():
        fields = [field.strip() for field in line.split('|')[1:-1]]
        if not fields or fields[0] == 'NPU' or all(field == 'N/A' for field in fields):
            continue
        if len(fields) >= 4 and re.fullmatch(r'\d+', fields[2]):
            if not fields[0].isdigit():
                raise RuntimeError(f'Unrecognized NPU ID: {line}')
            if int(fields[0]) in devices:
                pids.append(int(fields[2]))
        elif fields and any(fields):
            raise RuntimeError(f'Unrecognized context row: {line}')
    return pids


def validate_groups(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    mapping = {row['DEVICE']: (int(row['GROUP']), int(row['NPU ID'])) for row in rows}
    expected = {f'rbln{i}': (0, i) for i in range(4)}
    if mapping != expected:
        raise RuntimeError(f'Host benchmark requires physical ATOM0-3 in default RSD group 0: {mapping}')
    return mapping


def assert_atom_idle(directory):
    check = subprocess.run(['rbln-smi'], capture_output=True, text=True, timeout=15)
    (directory / 'preflight-rbln-smi.txt').write_text(check.stdout + check.stderr)
    try:
        if check.returncode:
            raise RuntimeError(f'rbln-smi exit {check.returncode}')
        pids = context_pids(check.stdout, devices=(2, 3))
        groups = Path('/sys/class/rebellions/rsd/rsd_group').read_text()
        (directory / 'preflight-rsd-groups.csv').write_text(groups)
        validate_groups(groups)
        if pids:
            raise RuntimeError(f'ATOM contexts already in use by PIDs {pids}')
    except Exception as exc:
        (directory / 'preflight.json').write_text(json.dumps(
            {'passed': False, 'error': str(exc), 'hardware_experiment_started': False}, indent=2))
        raise
    (directory / 'preflight.json').write_text(json.dumps({'passed': True, 'context_pids': []}))


def worker_environment(parent, device):
    env = dict(parent)
    env.pop('RBLN_DEVICE_MAP', None)
    env.pop('RBLN_DUMMY_DEVICE', None)
    env.update(RBLN_DEVICES=str(device), RBLN_NPUS_PER_DEVICE='1', PYTHONPATH=parent.get('PYTHONPATH',''))
    return env


def accept_worker(server, workers, timeout=300):
    """Watch startup only; this deadline is never used for an in-flight DMA."""
    deadline = time.monotonic() + timeout
    previous = server.gettimeout()
    try:
        server.settimeout(1)
        while True:
            failed = {device: proc.poll() for device, proc in workers.items() if proc.poll() is not None}
            if failed:
                raise RuntimeError(f'Worker exited during initialization: {failed}')
            if time.monotonic() >= deadline:
                raise TimeoutError('Worker registration deadline exceeded before DMA')
            try:
                return server.accept()
            except socket.timeout:
                continue
    finally:
        server.settimeout(previous)
