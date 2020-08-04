import unittest

from integrationutils.util.time_objects import TimeSpan

class Test_TimeSpan(unittest.TestCase):

    def test_constructor(self):
        ts = TimeSpan()
        self._check_timespan(ts, 0, 0)
        with self.assertRaises(ValueError):
            TimeSpan(-1, 0)
        with self.assertRaises(ValueError):
            TimeSpan(0, -1)
        ts = TimeSpan(123456789, 123456789)
        self._check_timespan(ts, 123456789, 123456789)
        ts = TimeSpan(123456789, 1234567891)
        self._check_timespan(ts, 123456790, 234567891)

    def test_from_seconds(self):
        with self.assertRaises(ValueError):
            TimeSpan.from_seconds(-1)
        ts = TimeSpan.from_seconds(1.1)
        self._check_timespan(ts, 1, 100000000)
        ts = TimeSpan.from_seconds(14853523.99999)
        self._check_timespan(ts, 14853523, 999990000)
        ts = TimeSpan.from_seconds(123456789.123456)
        self._check_timespan(ts, 123456789, 123456000)

    def test_to_seconds(self):
        ts = TimeSpan(123456789, 123456)
        self.assertTrue(ts.to_seconds() == 123456789.000123456)

    def _check_timespan(self, ts: TimeSpan, whole_seconds: int, nano_seconds: int):
        self.assertTrue(ts.whole_seconds == whole_seconds, 'Number of whole seconds not equal.')
        self.assertTrue(ts.nano_seconds == nano_seconds, 'Number of nanoseconds not equal.')

if __name__ == '__main__':
    unittest.main()
