"""Descriptor helpers for the HETERO_NPU NIXL backend (nixl_plugin/, docs/nixl-backend.md).

Stdlib only: these build the tuples that `nixl_agent.register_memory` / `get_xfer_descs` take.
The plugin itself does the transfer; nothing here touches a device.

    agent = nixl_agent('kv', nixl_agent_config(backends=[BACKEND]))
    agent.register_memory([rngd_region(src_ptr, n, '/dev/rngd/npu1ch0', 0)], 'VRAM', [BACKEND])
    agent.register_memory([dmabuf_region(fd, n, base=dva, dev_id=1)], 'VRAM', [BACKEND])
    h = agent.initialize_xfer('WRITE', agent.get_xfer_descs(src, 'VRAM'),
                              agent.get_xfer_descs(dst, 'VRAM'), agent.name, backends=[BACKEND])
"""
import re

BACKEND = 'HETERO_NPU'
ALIGN = 4096
_RNGD_PATH = re.compile(r'/dev/rngd/npu(0|[1-9][0-9]*)ch0')


def backend_params(queue_depth=1, desc_bytes=2 << 20, descs_per_cmd=8, align=ALIGN, allow_host=True):
    """Init params for `agent.create_backend(BACKEND, params)`; defaults = the native helper's scope."""
    if not 1 <= queue_depth <= 64 or desc_bytes <= 0 or desc_bytes % align or descs_per_cmd < 1:
        raise ValueError('queue_depth 1..64, desc_bytes a positive multiple of align, descs_per_cmd >= 1')
    return {'queue_depth': str(queue_depth), 'desc_bytes': str(desc_bytes),
            'descs_per_cmd': str(descs_per_cmd), 'align': str(align), 'allow_host': '1' if allow_host else '0'}


def rngd_region(address, size, device_path, dev_id):
    """VRAM registration of a live RNGD allocation. `address` is the raw DRAM offset (data_ptr)."""
    if not _RNGD_PATH.fullmatch(device_path):
        raise ValueError('canonical RNGD PE0 path /dev/rngd/npuNch0 required')
    if address <= 0 or address % 64 or size <= 0:
        raise ValueError('positive, 64-byte aligned RNGD address and positive size required')
    return (address, size, dev_id, f'rngd:{device_path}')


def dmabuf_region(fd, size, base, dev_id, offset=0):
    """VRAM registration of an exported DMA-BUF. Transfer addresses are `base + byte offset`."""
    if fd < 0 or size <= 0 or offset < 0 or base < 0:
        raise ValueError('valid fd, positive size, non-negative base/offset required')
    return (base, size, dev_id, f'dmabuf:{fd}:{offset}')


def host_region(address, size, dev_id=0):
    """DRAM registration of host memory (explicit staging leg, never a fallback)."""
    if address <= 0 or size <= 0:
        raise ValueError('positive host address and size required')
    return (address, size, dev_id, '')


def chunks(nbytes, count, align=ALIGN):
    """Split nbytes into `count` aligned (offset, length) pieces, e.g. one per layer.

    Every piece is a multiple of `align`; the first `remainder` pieces get one extra page.
    """
    if count < 1 or nbytes <= 0 or nbytes % align:
        raise ValueError('count >= 1 and a positive, aligned size required')
    pages = nbytes // align
    if count > pages:
        raise ValueError('more chunks than aligned pages')
    base, extra = divmod(pages, count)
    out, offset = [], 0
    for i in range(count):
        length = (base + (1 if i < extra else 0)) * align
        out.append((offset, length))
        offset += length
    return out


def map_range(regions, offset, length):
    """Transfer descriptors for logical bytes [offset, offset+length) of consecutive regions.

    `regions` is [(address, size, dev_id), ...] viewed as one buffer. Used when a buffer is larger
    than one device allocation (both SDKs cap a tensor below 4 GiB) and for layer-wise chunks.
    """
    if offset < 0 or length <= 0:
        raise ValueError('non-negative offset and positive length required')
    out, start, end = [], offset, offset + length
    for address, size, dev_id in regions:
        lo, hi = max(start, 0), min(end, size)
        if lo < hi:
            out.append((address + lo, hi - lo, dev_id))
        start -= size
        end -= size
        if end <= 0:
            break
    if end > 0:
        raise ValueError('range exceeds the regions')
    return out


def paired_ranges(src_regions, dst_regions, offset, length):
    """Equal-length (src, dst) descriptor lists for logical bytes [offset, offset+length).

    NIXL pairs the i-th local with the i-th remote descriptor, so pieces are cut wherever either
    side crosses a region boundary.
    """
    src, dst = map_range(src_regions, offset, length), map_range(dst_regions, offset, length)
    out_s, out_d = [], []
    while src:
        (sa, sn, sd), (da, dn, dd) = src[0], dst[0]
        n = min(sn, dn)
        out_s.append((sa, n, sd))
        out_d.append((da, n, dd))
        src[0], dst[0] = (sa + n, sn - n, sd), (da + n, dn - n, dd)
        if src[0][1] == 0:
            src.pop(0)
        if dst[0][1] == 0:
            dst.pop(0)
    return out_s, out_d
