import unittest
from hetero_p2p import nixl


class NixlDescriptorTests(unittest.TestCase):
    def test_regions_carry_backend_meta(self):
        self.assertEqual(nixl.rngd_region(0x178000000, 4096, '/dev/rngd/npu1ch0', 7),
                         (0x178000000, 4096, 7, 'rngd:/dev/rngd/npu1ch0'))
        self.assertEqual(nixl.dmabuf_region(5, 8192, base=0x10000000, dev_id=1),
                         (0x10000000, 8192, 1, 'dmabuf:5:0'))
        self.assertEqual(nixl.host_region(4096, 4096)[3], '')
        for bad in (lambda: nixl.rngd_region(4096, 4096, '/dev/rngd/npu1pe0', 0),
                    lambda: nixl.rngd_region(4097, 4096, '/dev/rngd/npu1ch0', 0),
                    lambda: nixl.dmabuf_region(-1, 4096, 0, 0),
                    lambda: nixl.host_region(0, 4096)):
            self.assertRaises(ValueError, bad)

    def test_chunks_are_aligned_and_cover_everything(self):
        pieces = nixl.chunks(100 * 4096, 32)
        self.assertEqual(len(pieces), 32)
        self.assertTrue(all(off % 4096 == 0 and n % 4096 == 0 and n > 0 for off, n in pieces))
        self.assertEqual(sum(n for _, n in pieces), 100 * 4096)
        self.assertEqual([o for o, _ in pieces][1:], [o + n for o, n in pieces][:-1])
        self.assertRaises(ValueError, nixl.chunks, 4096, 2)
        self.assertRaises(ValueError, nixl.chunks, 4097, 1)

    def test_map_range_crosses_region_boundaries(self):
        regions = [(1 << 30, 8192, 3), (5 << 30, 8192, 3)]
        self.assertEqual(nixl.map_range(regions, 0, 4096), [(1 << 30, 4096, 3)])
        self.assertEqual(nixl.map_range(regions, 4096, 8192),
                         [((1 << 30) + 4096, 4096, 3), (5 << 30, 4096, 3)])
        self.assertEqual(nixl.map_range(regions, 8192, 8192), [(5 << 30, 8192, 3)])
        self.assertRaises(ValueError, nixl.map_range, regions, 4096, 16384)

    def test_paired_ranges_cut_at_both_sides(self):
        src = [(1 << 30, 12288, 0)]
        dst = [(0, 4096, 1), (1 << 20, 8192, 1)]
        s, d = nixl.paired_ranges(src, dst, 0, 12288)
        self.assertEqual([n for _, n, _ in s], [4096, 8192])
        self.assertEqual([n for _, n, _ in d], [4096, 8192])
        self.assertEqual(s[1][0], (1 << 30) + 4096)
        self.assertEqual(d[1][0], 1 << 20)

    def test_backend_params_validate(self):
        self.assertEqual(nixl.backend_params()['queue_depth'], '1')
        self.assertEqual(nixl.backend_params(queue_depth=4, allow_host=False)['allow_host'], '0')
        self.assertRaises(ValueError, nixl.backend_params, queue_depth=0)
        self.assertRaises(ValueError, nixl.backend_params, desc_bytes=4097)


if __name__ == '__main__':
    unittest.main()
