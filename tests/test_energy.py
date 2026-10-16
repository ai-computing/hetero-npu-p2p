import unittest
from hetero_p2p.energy import integrate_power


def samples(points):
    return [dict(start=t,end=t,watts=w) for t,w in points]


class EnergyTests(unittest.TestCase):
    def test_constant_and_linear_boundary_interpolation(self):
        self.assertAlmostEqual(integrate_power(samples([(0,20),(1,20),(2,20)]),.25,1.75),30)
        self.assertAlmostEqual(integrate_power(samples([(0,0),(1,10),(2,20)]),.5,1.5),10)

    def test_missing_brackets_and_gaps_rejected(self):
        for points,lo,hi in [([(0,10),(1,10)],-.1,.5), ([(0,10),(1,10)],.5,1.1), ([(0,10),(4,10)],1,2)]:
            with self.assertRaises(ValueError):integrate_power(samples(points),lo,hi)

    def test_invalid_sensor_rows_rejected(self):
        for watts in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):integrate_power(samples([(0,10),(1,watts),(2,10)]),.5,1.5)
        with self.assertRaises(ValueError):integrate_power(samples([(0,10),(1,10),(1,10),(2,10)]),.5,1.5)
        rows=samples([(0,10),(2,10)])+[dict(start=.9,end=1.1,error='sensor unavailable')]
        with self.assertRaises(ValueError):integrate_power(rows,.5,1.5)

    def test_slow_command_rejected(self):
        rows=[dict(start=-1,end=1,watts=10),dict(start=1,end=3,watts=10)]
        with self.assertRaises(ValueError):integrate_power(rows,.5,1.5)
