import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mixed_sim', Path(__file__).resolve().parents[1]/'examples/koni/simulate_mixed_queue.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SimulationTests(unittest.TestCase):
    def test_full_equal_length_batches(self):
        for width in (12, 14):
            result = module.simulate([128]*84, width, 84, 'length', True)
            self.assertEqual(result['steps'], (84//width)*128)
            self.assertEqual(result['useful_lane_fraction'], 1)

    def test_fifo_refill_reuses_finished_lane(self):
        self.assertEqual(module.simulate([1,10,1],2,3,'fifo',False)['steps'],11)
        self.assertEqual(module.simulate([1,10,1],2,3,'fifo',True)['steps'],10)

    def test_tail_is_counted(self):
        result = module.simulate([2]*3,2,3,'fifo',True)
        self.assertEqual(result['steps'],4)
        self.assertEqual(result['useful_lane_fraction'],.75)
