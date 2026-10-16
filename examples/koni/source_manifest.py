"""Capture source provenance before model initialization; never collect credentials."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess


def capture_sources(output):
    root = Path(__file__).resolve().parents[2]
    repositories = {}
    for repo in (root, root.parent / 'rngd_llmd'):
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args])
        paths = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
        hashes = {}
        for raw in paths:
            if not raw:
                continue
            relative = raw.decode()
            path = repo / relative
            if path.suffix == '.py' and path.is_file() and not path.is_symlink():
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        repositories[repo.name] = dict(
            head=git('rev-parse', 'HEAD').decode().strip(),
            dirty=bool(git('status', '--porcelain')),
            files=hashes,
        )
    data = dict(capture_phase='startup_before_model_initialization',
                captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                repositories=repositories,
                scope='Python source on disk at startup; not a guarantee against subsequent external edits')
    # Exclusive creation prevents replacing provenance from a previous run.
    with (Path(output) / 'source-manifest.json').open('x') as stream:
        json.dump(data, stream, indent=2)
