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

    def test_from_minutes(self):
        A = TimeSpan.from_minutes(1.1)
        B = TimeSpan.from_minutes(1234567.99999)
        C = TimeSpan.from_minutes(-3.1415962)
        A_truth = TimeSpan(66, 0)
        B_truth = TimeSpan(74074079, 999399990)
        C_truth = TimeSpan(-188, -495772000)
        self.assertTrue(A == A_truth, f"TimeSpans are equal.")
        self.assertTrue(B == B_truth, f"TimeSpans are equal.")
        self.assertTrue(C == C_truth, f"TimeSpans are equal.")

    def test_from_hours(self):
        A = TimeSpan.from_hours(1.1)
        B = TimeSpan.from_hours(1234567.99999)
        C = TimeSpan.from_hours(-3.1415962)
        A_truth = TimeSpan(3960, 0)
        B_truth = TimeSpan(4444444799, 964000000)
        C_truth = TimeSpan(-11309, -746320000)
        self.assertTrue(A == A_truth, f"TimeSpans are equal.")
        self.assertTrue(B == B_truth, f"TimeSpans are equal.")
        self.assertTrue(C == C_truth, f"TimeSpans are equal.")

    def test_from_days(self):
        A = TimeSpan.from_days(1.1)
        B = TimeSpan.from_days(1234567.99999)
        C = TimeSpan.from_days(-3.1415962)
        A_truth = TimeSpan(95040, 0)
        B_truth = TimeSpan(106666675199, 135990000)
        C_truth = TimeSpan(-271433, -911680000)
        self.assertTrue(A == A_truth, f"TimeSpans are equal.")
        self.assertTrue(B == B_truth, f"TimeSpans are equal.")
        self.assertTrue(C == C_truth, f"TimeSpans are equal.")

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

    def test_abs(self):
        A = TimeSpan(-1,-1)
        B = TimeSpan(1,1)
        self.assertTrue(abs(A) == B, "Absolute value was not applied successfully.")

    def test_neg(self):
        A = TimeSpan(1,1)
        B = TimeSpan(-1,-1)
        self.assertTrue(-A == B, "Unary negation was not applied successfully.")

    def test_add(self):
        A = TimeSpan.from_days(1.0 + 2/24)
        B = TimeSpan.from_hours(2.0)
        C = TimeSpan.from_hours(28.0)
        D = TimeSpan.from_hours(3.0)
        E = TimeSpan.from_hours(25.0)
        self.assertTrue(A + B == C, "Failed to add TimeSpans properly.")
        self.assertTrue(B + A == C, "Failed to add TimeSpans properly.")
        self.assertTrue(A + B + (-D) == E, "Failed to add TimeSpans properly.")
        self.assertTrue(C + (-A) + (-B) == TimeSpan.zero(), "Failed to add TimeSpans properly.")

    def test_subract(self):
        A = TimeSpan.from_days(1.0 + 2/24)
        B = TimeSpan.from_hours(2.0)
        C = TimeSpan.from_hours(28.0)
        D = TimeSpan.from_hours(3.0)
        E = TimeSpan.from_hours(25.0)
        self.assertTrue(C - A == B, "Failed to subtract TimeSpans properly.")
        self.assertTrue(A - C == -B, "Failed to subtract TimeSpans properly.")
        self.assertTrue(C - A - B == TimeSpan.zero(), "Failed to subtract TimeSpans properly.")
        self.assertTrue(C - D == E, "Failed to subtract TimeSpans properly.")

    def test_mult(self):
        A = TimeSpan.from_hours(2.0)
        B = TimeSpan.from_hours(6.0)
        C = TimeSpan.from_hours(3.0)
        D = TimeSpan.from_hours(-11.0)
        self.assertTrue(3 * A == B, "Failed to multiply TimeSpan properly.")
        self.assertTrue(1.5 * A == C, "Failed to multiply TimeSpan properly.")
        self.assertTrue(-(11/2) * A == D, "Failed to multiply TimeSpan properly.")

    def test_to_seconds(self):
        A = TimeSpan(123456789, 123456)
        B = TimeSpan(-1, -550000000)
        self.assertTrue(A.to_seconds() == 123456789.000123456)
        self.assertTrue(B.to_seconds() == -1.55)

    def test_to_minutes(self):
        A = TimeSpan(123456789, 123456)
        B = TimeSpan(-1, -550000000)
        self.assertTrue(A.to_minutes() == 2057613.1500020577)
        self.assertTrue(B.to_minutes() == -0.025833333333333333)

    def test_to_hours(self):
        A = TimeSpan(123456789, 123456)
        B = TimeSpan(-1, -550000000)
        self.assertTrue(A.to_hours() == 34293.5525000343)
        self.assertTrue(B.to_hours() == -0.00043055555555555555)

    def test_to_days(self):
        A = TimeSpan(123456789, 123456)
        B = TimeSpan(-1, -550000000)
        self.assertTrue(A.to_days() == 1428.8980208347623)
        self.assertTrue(B.to_days() == -1.7939814814814815e-05)

if __name__ == '__main__':
    unittest.main()
