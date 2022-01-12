""" Module contains unit test definitions for the astrolib.base_objects package.
"""
import math
import unittest

from astrolib.base_objects import Matrix
from astrolib.base_objects import Vector3
from astrolib.base_objects import TimeSpan

#pylint: disable=invalid-name
#pylint: disable=line-too-long
#pylint: disable=missing-class-docstring
#pylint: disable=missing-function-docstring
#pylint: disable=pointless-statement
#pylint: disable=protected-access



class Test_Matrix(unittest.TestCase):

    def test_constructor(self):
        A = Matrix([[1,2,3],[4,5,6]])
        for i in range(A.num_rows):
            for j in range(A.num_cols):
                self.assertTrue(A[i,j] == i*3+j+1, "Matrix construction not done correctly.")
        with self.assertRaises(ValueError):
            Matrix([[1,2],[3,4,5]])

    def test_get_item(self):
        A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        B = Matrix([[4,5,6]])
        C = Matrix([[2],[5],[8]])
        D = Matrix([[4,6]])
        E = Matrix([[2],[8]])
        F = Matrix([[1,3],[7,9]])
        G = Matrix([[2], [5]])
        H = Matrix([[5, 6]])
        self.assertTrue(A[1,1] == 5, "Matrix indexing not done correctly")
        self.assertTrue(A[1,:] == B, "Matrix indexing not done correctly")
        self.assertTrue(A[:,1] == C, "Matrix indexing not done correctly")
        self.assertTrue(A[1,0:3:2] == D, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2,1] == E, "Matrix indexing not done correctly")
        self.assertTrue(A[:,:] == A, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2,0:3:2] == F, "Matrix indexing not done correctly")
        self.assertTrue(C[1] == 5, "Matrix indexing not done correctly.")
        self.assertTrue(C[0:2] == G, "Matrix indexing not done correctly.")
        self.assertTrue(B[1:3] == H, "Matrix indexing not done correctly.")
        with self.assertRaises(ValueError):
            A[0:2]

    def test_set_item(self):
        A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        B = Matrix([[1,9,3],[4,5,6],[7,8,9]])
        C = Matrix([[1,3,3],[4,6,6],[7,9,9]])
        D = Matrix([[1,1,1],[4,6,6],[7,9,9]])
        E = Matrix([[0,9,0],[4,5,6],[0,8,0]])
        F = Matrix([[1,9,1],[1,5,1],[1,8,1]])
        G = Matrix([[1,1],[1,1],[1,1]])
        H = Matrix([[1,1],[1,1]])
        A[0,1] = 9
        self.assertTrue(A == B, "Matrix assignment not done correctly")
        A[:,1] = Matrix([[3],[6],[9]])
        self.assertTrue(A == C, "Matrix assignment not done correctly")
        A[0,:] = Matrix.ones(1,3)
        self.assertTrue(A == D, "Matrix assignment not done correctly")
        A[:,:] = B
        self.assertTrue(A == B, "Matrix assignment not done correctly")
        A[0:3:2,0:3:2] = Matrix.zeros(2)
        self.assertTrue(A == E, "Matrix assignment not done correctly")
        A[:,0:3:2] = Matrix.ones(3,2)
        self.assertTrue(A == F, "Matrix assignment not done correctly")
        with self.assertRaises(ValueError):
            A[:,:] = 1
        with self.assertRaises(ValueError):
            A[1:2,:] = 1
        with self.assertRaises(ValueError):
            A[:,1:2] = 1
        with self.assertRaises(ValueError):
            A[1,1] = B
        A[:,1] = Matrix.empty()
        self.assertTrue(A == G, "Matrix assignment not done correctly")
        A[0,:] = Matrix.empty()
        self.assertTrue(A == H, "Matrix assignment not done correctly")
        with self.assertRaises(ValueError):
            A[:,:] = Matrix.empty()

    def test_num_rows(self):
        A = Matrix([[1,2,3],[4,5,6]])
        self.assertTrue(A.num_rows == 2)
        with self.assertRaises(AttributeError):
            A.num_rows = 1

    def test_num_cols(self):
        A = Matrix([[1,2,3],[4,5,6]])
        self.assertTrue(A.num_cols == 3)
        with self.assertRaises(AttributeError):
            A.num_cols = 1

    def test_size(self):
        A = Matrix([[1,2,3],[4,5,6]])
        self.assertTrue(A.size == (2,3))
        with self.assertRaises(AttributeError):
            A.size = 1

    def test_is_empty(self):
        A = Matrix([])
        B = Matrix([[1],[2]])
        C = Matrix.empty()
        self.assertTrue(A.is_empty, "Matrix is empty.")
        self.assertTrue(not B.is_empty, "Matrix is not empty.")
        self.assertTrue(C.is_empty, "Matrix is empty.")

    def test_fill(self):
        A = Matrix.fill(2, 2, 1.0)
        B = Matrix([[1,1],[1,1]])
        for i in range(2):
            for j in range(2):
                self.assertTrue(A[i,j] == B[i,j], f"Matrix element [{i},{j}] not equal.")
        with self.assertRaises(ValueError):
            Matrix.fill(-1,2,1)
        with self.assertRaises(ValueError):
            Matrix.fill(0,2,1)
        with self.assertRaises(ValueError):
            Matrix.fill(2,-1,1)
        with self.assertRaises(ValueError):
            Matrix.fill(2,0,1)

    def test_zeros(self):
        A = Matrix.zeros(2,3)
        B = Matrix([[0,0,0],[0,0,0]])
        C = Matrix.zeros(3)
        D = Matrix([[0,0,0],[0,0,0],[0,0,0]])
        for i in range(2):
            for j in range(3):
                self.assertTrue(A[i,j] == B[i,j], f"Matrix element [{i},{j}] not equal.")
        for i in range(3):
            for j in range(3):
                self.assertTrue(C[i,j] == D[i,j], f"Matrix element [{i},{j}] not equal.")
        with self.assertRaises(ValueError):
            Matrix.zeros(-1,2)
        with self.assertRaises(ValueError):
            Matrix.zeros(0,2)
        with self.assertRaises(ValueError):
            Matrix.zeros(2,-1)
        with self.assertRaises(ValueError):
            Matrix.zeros(2,0)

    def test_ones(self):
        A = Matrix.ones(2,3)
        B = Matrix([[1,1,1],[1,1,1]])
        C = Matrix.ones(3)
        D = Matrix([[1,1,1],[1,1,1],[1,1,1]])
        for i in range(2):
            for j in range(3):
                self.assertTrue(A[i,j] == B[i,j], f"Matrix element [{i},{j}] not equal.")
        for i in range(3):
            for j in range(3):
                self.assertTrue(C[i,j] == D[i,j], f"Matrix element [{i},{j}] not equal.")
        with self.assertRaises(ValueError):
            Matrix.ones(-1,2)
        with self.assertRaises(ValueError):
            Matrix.ones(0,2)
        with self.assertRaises(ValueError):
            Matrix.ones(2,-1)
        with self.assertRaises(ValueError):
            Matrix.ones(2,0)

    def test_identity(self):
        A = Matrix.identity(2)
        B = Matrix([[1,0],[0,1]])
        C = Matrix.identity(3)
        D = Matrix([[1,0,0],[0,1,0],[0,0,1]])
        for i in range(2):
            for j in range(2):
                self.assertTrue(A[i,j] == B[i,j], f"Matrix element [{i},{j}] not equal.")
        for i in range(3):
            for j in range(3):
                self.assertTrue(C[i,j] == D[i,j], f"Matrix element [{i},{j}] not equal.")
        with self.assertRaises(ValueError):
            Matrix.identity(-1)
        with self.assertRaises(ValueError):
            Matrix.identity(0)

    def test_from_column_matrices(self):
        A = Matrix.zeros(3,1)
        B = Matrix.ones(4,1)
        C = Matrix.fill(2,1,1.2345)
        D = Vector3(1,2,3)
        E = Vector3.zeros()
        F = Matrix.zeros(2,2)
        G = Matrix.from_column_matrices([D,E])
        truth_mat_1 = Matrix([[0],[0],[0],[1],[1],[1],[1],[1.2345],[1.2345]])
        truth_mat_2 = Matrix([[1],[2],[3],[0],[0],[0]])
        truth_mat_3 = Matrix([[0],[0],[0],[1],[1],[1],[1],[1.2345],[1.2345],[1],[2],[3],[0],[0],[0]])
        self.assertTrue(Matrix.from_column_matrices([A,B,C]) == truth_mat_1, "The matrix instantiation from column matrices was not completed successfully.")
        for row in G:
            self.assertTrue(isinstance(row[0], (float,int)), "The matrix instantiation from column matrices was not completed successfully.")
        self.assertTrue(G == truth_mat_2, "The matrix instantiation from column matrices was not completed successfully.")
        self.assertTrue(Matrix.from_column_matrices([A,B,C,D,E]) == truth_mat_3, "The matrix instantiation from column matrices was not completed successfully.")
        with self.assertRaises(ValueError):
            Matrix.from_column_matrices([A,F])

    def test_equals(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[4,5,6],[7,8,9]])
        C = Matrix([[1,2,3],[4,5,6]])
        D = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        E = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        F = 1.0
        self.assertTrue(A != B, "Matrices are not equal.")
        self.assertTrue(A == C, "Matrices are equal.")
        self.assertTrue(A != D, "Matrices are not equal.")
        self.assertTrue(A != E, "Matrices are not equal.")
        self.assertTrue(A != F, "Matrices are not equal.")
        #pylint: disable=not-an-iterable
        self.assertTrue(Matrix.ones(*A.size) == F, "Matrices are equal.")

    def test_neg(self):
        A = Matrix.ones(3,2)
        B = Matrix.fill(3,2,-1)
        self.assertTrue(-A == B, "Matrices are equal")

    def test_add(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[4,5,6],[7,8,9]])
        C = Matrix([[5,7,9],[11,13,15]])
        D = Matrix([[-1,-2,-3],[-4,-5,-6]])
        E = Matrix.ones(1, 1)
        self.assertTrue(A + B == C, "The matrix sum was not computed successfully.")
        self.assertTrue(B + A == C, "The matrix sum was not computed successfully.")
        self.assertTrue(A + D == Matrix.zeros(2,3), "The matrix sum was not computed successfully.")
        self.assertTrue(D + A == Matrix.zeros(2,3), "The matrix sum was not computed successfully.")
        self.assertTrue(A + 3 == B, "The matrix sum was not computed successfully.")
        self.assertTrue(3 + A == B, "The matrix sum was not computed successfully.")
        with self.assertRaises(TypeError):
            A + "foo"
        with self.assertRaises(TypeError):
            "foo" + A
        with self.assertRaises(ValueError):
            A + E
        with self.assertRaises(ValueError):
            E + A

    def test_subtract(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[4,5,6],[7,8,9]])
        C = Matrix([[-2,-1,0],[1,2,3]])
        D = Matrix([[2,1,0],[-1,-2,-3]])
        self.assertTrue(A - B == Matrix.fill(2,3,-3), "The matrix difference was not computed successfully.")
        self.assertTrue(B - A == Matrix.fill(2,3,3), "The matrix difference was not computed successfully.")
        self.assertTrue(A - 3 == C, "The matrix difference was not computed successfully.")
        self.assertTrue(3 - A == D, "The matrix difference was not computed successfully.")
        with self.assertRaises(TypeError):
            A - "foo"
        with self.assertRaises(TypeError):
            "foo" - A

    def test_mult(self):
        A = Matrix([[4,4,3,1,2],[2,9,8,1,6],[3,6,8,6,5],[7,6,4,8,1],[5,10,6,10,4]])
        B = Matrix([[0.1206,0.2518,0.9827,0.9063,0.0225],[0.5895,0.2904,0.7302,0.8797,0.4253],[0.2262,0.6171,0.3439,0.8178,0.3127],[0.3846,0.2653,0.5841,0.2607,0.1615],[0.5830,0.8244,0.1078,0.5944,0.1788]])
        C = Matrix([[5.0696,5.9343,8.6829,11.0466,3.2483],[11.2388,13.2658,12.5193,20.0984,7.6082],[10.9310,13.1484,14.1238,19.0751,6.9836],[8.9460,8.9203,17.4160,17.5733,5.4307],[14.0334,13.8163,20.5508,23.2193,8.5714]])
        D = Matrix([[10.3908,14.3077,13.9979,13.7440,7.6617],[13.4135,18.8840,16.0042,16.5513,9.1536],[10.4585,16.5556,13.5137,12.5758,7.9429],[6.4538,10.6096,9.9605,7.8550,6.1879],[9.3583,15.7517,12.6561,8.5965,7.9605]])
        E = Matrix([[31.1723,42.9230,41.9937,41.2320,22.9852],[40.2406,56.6520,48.0126,49.6538,27.4608],[31.3754,49.6667,40.5410,37.7274,23.8288],[19.3613,31.8289,29.8814,23.5650,18.5638],[28.0750,47.2552,37.9684,25.7895,23.8815]])
        F = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        G = Matrix([[1,2],[3,4],[5,6]])
        H = Matrix([[22,28],[49,64],[76,100]])
        tol = 2.0e-3
        self.assertTrue(abs(A * B - C) <= tol, "The matrix product was not computed successfully.")
        self.assertTrue(abs(B * A - D) <= tol, "The matrix product was not computed successfully.")
        self.assertTrue(abs(3 * D - E) <= tol, "The matrix product was not computed successfully.")
        self.assertTrue(abs(D * 3 - E) <= tol, "The matrix product was not computed successfully.")
        self.assertTrue(abs(F * G - H) <= tol, "The matrix product was not computed successfully.")
        with self.assertRaises(ValueError):
            G * F

    def test_transpose(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[1,4],[2,5],[3,6]])
        self.assertTrue(A.transpose() == B, "The matrix transpose was not computed successfully.")

    def test_determinant(self):
        A = Matrix([[3,0,1],[0,5,0],[-1,1,-1]])
        B = Matrix([[3,0,1],[0,5,0]])
        self.assertTrue(A.determinant() == -10, "The matrix determinant was not computed successfully.")
        with self.assertRaises(ValueError):
            B.determinant()
        # TODO Add more test cases

    def test_inverse(self):
        A = Matrix([[3,0,1],[0,5,0],[-1,1,-1]])
        B = Matrix([[0.5,-0.1,0.5],[0,0.2,0],[-0.5,0.3,-1.5]])
        C = Matrix([[3,0,1],[0,5,0]])
        tol = 1.0e-10
        self.assertTrue(A.inverse() - B <= Matrix.fill(A.num_rows, A.num_cols, tol), "The matrix inverse was not computed successfully.")
        with self.assertRaises(ValueError):
            C.inverse()
        # TODO Add more test cases


class Test_Vector3(unittest.TestCase):

    def test_constructor(self):
        A = Vector3(1,2,3)
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 2, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 3, "The vector was not initialized successfully.")
        B = Vector3()
        self.assertTrue(B.x == 0, "The vector was not initialized successfully.")
        self.assertTrue(B.y == 0, "The vector was not initialized successfully.")
        self.assertTrue(B.z == 0, "The vector was not initialized successfully.")

    def test_ones(self):
        A = Vector3.ones()
        self.assertTrue(A.x == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 1, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 1, "The vector was not initialized successfully.")

    def test_zeros(self):
        A = Vector3.zeros()
        self.assertTrue(isinstance(A.x, int), "The vector was not initialized successfully.")
        self.assertTrue(isinstance(A.y, int), "The vector was not initialized successfully.")
        self.assertTrue(isinstance(A.z, int), "The vector was not initialized successfully.")
        self.assertTrue(A.x == 0, "The vector was not initialized successfully.")
        self.assertTrue(A.y == 0, "The vector was not initialized successfully.")
        self.assertTrue(A.z == 0, "The vector was not initialized successfully.")

    def test_add(self):
        A = Vector3(1,2,3)
        B = Vector3(4,5,6)
        C = Vector3(5,7,9)
        self.assertTrue(A + B == C, "The vector sum was not computed successfully.")
        self.assertTrue(B + A == C, "The vector sum was not computed successfully.")
        self.assertTrue(A + 3 == B, "The vector sum was not computed successfully.")
        self.assertTrue(3 + A == B, "The vector sum was not computed successfully.")

    def test_subtract(self):
        A = Vector3(1,2,3)
        B = Vector3(4,5,6)
        C = Vector3(5,7,9)
        self.assertTrue(C - B == A, "The vector difference was not computed successfully.")
        self.assertTrue(C - A == B, "The vector difference was not computed successfully.")
        self.assertTrue(B - 3 == A, "The vector difference was not computed successfully.")

    def test_mul_with_matrix(self):
        x = Vector3(1,2,3)
        A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Vector3(14,32,50)
        self.assertTrue(Matrix.identity(3) * x == x, "The matrix multiplication with the vector was not computed successfully.")
        self.assertTrue(A * x == b, "The matrix multiplication with the vector was not computed successfully.")

    def test_norm(self):
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(A.norm() - 3.0 <= 1.0e-16, "The vector norm was not computed successfully.")
        self.assertTrue(A.norm_2() - 9.0 <= 1.0e-16, "The vector norm was not computed successfully.")

    def test_cross(self):
        #TODO Update Matrix class to utilize Decimal class for its elements instead of floats to increase numeric precision
        A = Vector3(0.5377, 1.8339, -2.2588)
        B = Vector3(3.0349, 0.7254, -0.0631)
        AxB = Vector3(1.522941669438591, -6.821524812007696, -5.175674650707535)
        BxA = Vector3(-1.522941669438591, 6.821524812007696, 5.175674650707535)
        self.assertTrue(abs(A.cross(B) - AxB) <= 1.0e-03 * Vector3.ones(), "The cross product was not computed successfully.")
        self.assertTrue(abs(B.cross(A) - BxA) <= 1.0e-03 * Vector3.ones(), "The cross product was not computed successfully.")

    def test_dot(self):
        #TODO Update Matrix class to utilize Decimal class for its elements instead of floats to increase numeric precision
        A = Vector3(0.5377, 1.8339, -2.2588)
        B = Vector3(3.0349, 0.7254, -0.0631)
        dot_product = 3.104517858912047
        self.assertTrue(abs(A.dot(B) - dot_product) <= 1.0e-03, "The dot product was not computed successfully.")
        self.assertTrue(abs(B.dot(A) - dot_product) <= 1.0e-03, "The dot product was not computed successfully.")

    def test_normalize(self):
        #TODO Update Matrix class to utilize Decimal class for its elements instead of floats to increase numeric precision
        A = Vector3(0.5377, 1.8339, -2.2588)
        A_norm = Vector3(0.1792, 0.6113, -0.7529)
        self.assertTrue(abs(A.normalized() - A_norm) <= 1.0e-01 * Vector3.ones(), "The normalized form of the vector was not computed successfully.")
        A.normalize()
        self.assertTrue(abs(A - A_norm) <= 1.0e-01 * Vector3.ones(), "The normalized form of the vector was not computed successfully.")
        self.assertTrue(A.norm() == 1.0, "The normalized form of the vector was not computed successfully.")

    def test_neg(self):
        A = Vector3(0.5377, 1.8339, -2.2588)
        self.assertTrue(isinstance(-A, Vector3))
        self.assertTrue(abs(A + -A) <= 1.0e-4)

    def test_vertex_angle(self):
        A = Vector3(1, 0, 0)
        B = Vector3(0, 1, 0)
        C = Vector3(math.cos(math.pi/6), math.sin(math.pi/6), 0)
        self.assertTrue(abs(A.vertex_angle(B) - math.pi/2) <= 1.0e-6)
        self.assertTrue(abs(A.vertex_angle(C) - math.pi/6) <= 1.0e-6)

class Test_TimeSpan(unittest.TestCase):

    def test_constructor(self):
        A = TimeSpan(1,2)
        B = TimeSpan(123456789, 123456789)
        C = TimeSpan(123456789, 1234567891)
        D = TimeSpan(-1, -2.1e9)
        #TODO Add tests for automagic handling of +/- combinations of whole and nano seconds
        self.assertTrue(A._whole_seconds == 1, 'Number of whole seconds not equal.')
        self.assertTrue(A._nano_seconds == 2, 'Number of nanoseconds not equal.')
        self.assertTrue(B._whole_seconds == 123456789, 'Number of whole seconds not equal.')
        self.assertTrue(B._nano_seconds == 123456789, 'Number of nanoseconds not equal.')
        self.assertTrue(C._whole_seconds == 123456790, 'Number of whole seconds not equal.')
        self.assertTrue(C._nano_seconds == 234567891, 'Number of nanoseconds not equal.')
        self.assertTrue(D._whole_seconds == -3, 'Number of whole seconds not equal.')
        self.assertTrue(D._nano_seconds == -100000000, 'Number of nanoseconds not equal.')

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
        self.assertTrue(A == A_truth, "TimeSpans are equal.")
        self.assertTrue(B == B_truth, "TimeSpans are equal.")
        self.assertTrue(C == C_truth, "TimeSpans are equal.")

    def test_from_hours(self):
        A = TimeSpan.from_hours(1.1)
        B = TimeSpan.from_hours(1234567.99999)
        C = TimeSpan.from_hours(-3.1415962)
        A_truth = TimeSpan(3960, 0)
        B_truth = TimeSpan(4444444799, 964000000)
        C_truth = TimeSpan(-11309, -746320000)
        self.assertTrue(A == A_truth, "TimeSpans are equal.")
        self.assertTrue(B == B_truth, "TimeSpans are equal.")
        self.assertTrue(C == C_truth, "TimeSpans are equal.")

    def test_from_days(self):
        A = TimeSpan.from_days(1.1)
        B = TimeSpan.from_days(1234567.99999)
        C = TimeSpan.from_days(-3.1415962)
        A_truth = TimeSpan(95040, 0)
        B_truth = TimeSpan(106666675199, 135990000)
        C_truth = TimeSpan(-271433, -911680000)
        self.assertTrue(A == A_truth, "TimeSpans are equal.")
        self.assertTrue(B == B_truth, "TimeSpans are equal.")
        self.assertTrue(C == C_truth, "TimeSpans are equal.")

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
        self.assertTrue(A >= A, "Conditional was not met properly.")
        self.assertTrue(A <= A, "Conditional was not met properly.")
        self.assertTrue(A <= A, "Conditional was not met properly.")
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

    def test_is_defined(self):
        A = TimeSpan.undefined()
        B = TimeSpan.zero()
        self.assertTrue(not A.is_defined(), "Defined-ness check failed.")
        self.assertTrue(B.is_defined(), "Defined-ness check failed.")


if __name__ == '__main__':
    unittest.main()
