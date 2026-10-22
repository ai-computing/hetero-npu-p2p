"""Per-pair path table: which device-to-device path exists between two memory kinds, who initiates it and what the
host must provide. Lookups never pick a substitute: a missing path raises PathUnavailable, and host staging is
never offered as a fallback (rm-core PROTOCOL §4.5).

Memory kinds: 'rngd' (Furiosa RNGD DRAM), 'atom' (Rebellions ATOM, exported DMA-BUF), 'cuda' (NVIDIA GPU memory),
'host'. Evidence: RNGD -> ATOM is the validated KV path of this package; GPU <-> RNGD is #6 / #7.
"""
from dataclasses import dataclass
from pathlib import Path as _FsPath

KINDS = ('rngd', 'atom', 'cuda', 'host')
NVIDIA_VERSION_FILE = '/proc/driver/nvidia/version'


class PathUnavailable(RuntimeError):
    """No device-to-device path for this pair on this host. Nothing was submitted and nothing falls back."""


class ProprietaryModuleError(PathUnavailable):
    """GPU <-> RNGD needs CUDA DMA-BUF export, which only the NVIDIA open kernel module provides."""


@dataclass(frozen=True)
class P2PPath:
    src: str
    dst: str
    status: str            # 'supported' | 'blocked' | 'unvalidated'
    initiator: str = ''    # device whose DMA engine moves the bytes
    mechanism: str = ''
    backend: str = ''      # hetero_p2p class that runs it
    requires: tuple = ()   # host prerequisites checked by resolve()
    reason: str = ''


TABLE = {p.src + '>' + p.dst: p for p in (
    P2PPath('rngd', 'atom', 'supported', 'rngd', 'RNGD PDMA DMABUF_READ into an ATOM-exported DMA-BUF',
            'RngdPdmaBackend', ('atom-export',)),
    P2PPath('rngd', 'cuda', 'supported', 'rngd', 'RNGD PDMA DMABUF_READ into CUDA memory exported as DMA-BUF',
            'RngdDmabufBackend', ('nvidia-open-module',)),
    P2PPath('cuda', 'rngd', 'supported', 'rngd', 'RNGD PDMA DMABUF_WRITE from CUDA memory exported as DMA-BUF',
            'RngdDmabufBackend', ('nvidia-open-module',),
            'the GPU cannot initiate it: cudaHostRegisterIoMemory on the RNGD BAR fails with cudaError 800 (#6)'),
    P2PPath('cuda', 'atom', 'blocked', reason='no host with both devices has been validated (#6)'),
    P2PPath('atom', 'cuda', 'blocked', reason='no host with both devices; ATOM exposes no public DMA initiation (#6)'),
    P2PPath('atom', 'rngd', 'unvalidated', 'rngd', 'RNGD PDMA DMABUF_WRITE from an ATOM-exported DMA-BUF',
            reason='plausible with the same driver command but not measured'),
    P2PPath('rngd', 'rngd', 'unvalidated', reason='not measured yet (#6)'),
    P2PPath('cuda', 'cuda', 'blocked', reason='out of scope: use NCCL or CUDA IPC'),
)}


def nvidia_kernel_module(version_file=None):
    """'open', 'proprietary', or None when no NVIDIA driver is loaded (from /proc/driver/nvidia/version)."""
    try:
        text = _FsPath(version_file or NVIDIA_VERSION_FILE).read_text()
    except OSError:
        return None
    return 'open' if 'Open Kernel Module' in text else 'proprietary'


def require_nvidia_open_module(version_file=None):
    flavour = nvidia_kernel_module(version_file)
    if flavour == 'open':
        return
    if flavour is None:
        raise PathUnavailable('GPU <-> RNGD: no NVIDIA kernel driver loaded')
    raise ProprietaryModuleError(
        'GPU <-> RNGD needs CUDA memory exported as DMA-BUF, which only the NVIDIA open kernel module provides '
        '(e.g. nvidia-driver-<version>-server-open); the proprietary module is loaded. No host-staged fallback is '
        'used.')


_CHECKS = {'nvidia-open-module': require_nvidia_open_module}


def lookup(src, dst):
    """The table entry for src -> dst (any status); KeyError for unknown kinds."""
    for kind in (src, dst):
        if kind not in KINDS:
            raise KeyError(f'unknown memory kind {kind!r}; one of {KINDS}')
    if 'host' in (src, dst):
        return P2PPath(src, dst, 'blocked', reason='host memory is a staging leg, never a device-to-device path; '
                                                    'stage explicitly if you need it')
    return TABLE.get(src + '>' + dst) or P2PPath(src, dst, 'blocked', reason='no known path')


def resolve(src, dst, check=True):
    """The supported path for src -> dst, with its host prerequisites checked; PathUnavailable otherwise."""
    path = lookup(src, dst)
    if path.status != 'supported':
        raise PathUnavailable(f'{src} -> {dst}: {path.status}: {path.reason}')
    if check:
        for need in path.requires:
            fn = _CHECKS.get(need)
            if fn:
                fn()
    return path
