import unittest

from integrationutils.util.time_objects import TimeSpan

class Test_TimeSpan(unittest.TestCase):

    def test_constructor(self):
        A = TimeSpan(1,2)
        B = TimeSpan(123456789, 123456789)
        C = TimeSpan(123456789, 1234567891)
        D = TimeSpan(-1, -2.1e9)
        #TODO Add tests for automagic handling of +/- combinations of whole and nano seconds
        self.assertTrue(A.whole_seconds == 1, 'Number of whole seconds not equal.')
        self.assertTrue(A.nano_seconds == 2, 'Number of nanoseconds not equal.')
        self.assertTrue(B.whole_seconds == 123456789, 'Number of whole seconds not equal.')
        self.assertTrue(B.nano_seconds == 123456789, 'Number of nanoseconds not equal.')
        self.assertTrue(C.whole_seconds == 123456790, 'Number of whole seconds not equal.')
        self.assertTrue(C.nano_seconds == 234567891, 'Number of nanoseconds not equal.')
        self.assertTrue(D.whole_seconds == -3, 'Number of whole seconds not equal.')
        self.assertTrue(D.nano_seconds == -100000000, 'Number of nanoseconds not equal.')

    def test_from_seconds(self):
        A = TimeSpan.from_seconds(1.1)
        B = TimeSpan.from_seconds(14853523.99999)
        C = TimeSpan.from_seconds(123456789.123456)
        D = TimeSpan.from_seconds(-3.1415962)
        A_truth = TimeSpan(1, 100000000)
        B_truth = TimeSpan(14853523, 999990000)
        C_truth = TimeSpan(123456789, 123456000)
        D_truth = TimeSpan(-3, -141596200)
        self.assertTrue(A == A_truth, "TimeSpans are equal.")
        self.assertTrue(B == B_truth, "TimeSpans are equal.")
        self.assertTrue(C == C_truth, "TimeSpans are equal.")
        self.assertTrue(D == D_truth, "TimeSpans are equal.")

    def test_equals(self):
        A = TimeSpan(1,1)
        B = TimeSpan(1,2)
        C = TimeSpan(2,1)
        D = TimeSpan.undefined()
        E = TimeSpan.from_seconds(1.23456)
        self.assertTrue(A == A, "TimeSpans are equal.")
        self.assertTrue(A != B, "TimeSpans are not equal.")
        self.assertTrue(A != C, "TimeSpans are not equal.")
        self.assertTrue(A != D, "TimeSpans are not equal.")
        self.assertTrue(A != E, "TimeSpans are not equal.")
        self.assertTrue(D == D, "TimeSpans are equal.")

    def test_conditionals(self):
        A = TimeSpan(60,0)
        B = TimeSpan(61,0)
        C = TimeSpan(60,1)
        D = TimeSpan(60,2)
        E = TimeSpan(59,9999)
        self.assertTrue(not A < A, "Conditional was not met properly.")
        self.assertTrue(A <= A, "Conditional was not met properly.")
        self.assertTrue(not A > A, "Conditional was not met properly.")
        self.assertTrue(A >= A, "Conditional was not met properly.")
        self.assertTrue(A < B, "Conditional was not met properly.")
        self.assertTrue(B > A, "Conditional was not met properly.")
        self.assertTrue(A < C, "Conditional was not met properly.")
        self.assertTrue(C < D, "Conditional was not met properly.")
        self.assertTrue(A > E, "Conditional was not met properly.")

    def test_add(self):
        pass

    def test_subract(self):
        pass

    def test_mult(self):
        pass

    def test_to_seconds(self):
        A = TimeSpan(123456789, 123456)
        B = TimeSpan(-1, -550000000)
        self.assertTrue(A.to_seconds() == 123456789.000123456)
        self.assertTrue(B.to_seconds() == -1.55)

if __name__ == '__main__':
    unittest.main()
