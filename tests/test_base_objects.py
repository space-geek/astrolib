import unittest

from integrationutils.util.base_objects import Matrix
from integrationutils.util.base_objects import Vec3d

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
        self.assertTrue(A[1,1] == 5, "Matrix indexing not done correctly")
        self.assertTrue(A[1,:] == B, "Matrix indexing not done correctly")
        self.assertTrue(A[:,1] == C, "Matrix indexing not done correctly")
        self.assertTrue(A[1,0:3:2] == D, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2,1] == E, "Matrix indexing not done correctly")
        self.assertTrue(A[:,:] == A, "Matrix indexing not done correctly")
        self.assertTrue(A[0:3:2,0:3:2] == F, "Matrix indexing not done correctly")

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

    def test_equals(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[4,5,6],[7,8,9]])
        C = Matrix([[1,2,3],[4,5,6]])
        D = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        E = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        self.assertTrue(A != B, "Matrices are not equal.")
        self.assertTrue(A == C, "Matrices are equal.")
        self.assertTrue(A != D, "Matrices are not equal.")
        self.assertTrue(A != E, "Matrices are not equal.")

    def test_add(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[4,5,6],[7,8,9]])
        C = Matrix([[5,7,9],[11,13,15]])
        D = Matrix([[-1,-2,-3],[-4,-5,-6]])
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
        self.assertTrue(abs(A * B - C) <= Matrix.fill(C.num_rows, C.num_cols, tol), "The matrix product was not computed successfully.")
        self.assertTrue(abs(B * A - D) <= Matrix.fill(D.num_rows, D.num_cols, tol), "The matrix product was not computed successfully.")
        self.assertTrue(abs(3 * D - E) <= Matrix.fill(E.num_rows, E.num_cols, tol), "The matrix product was not computed successfully.")
        self.assertTrue(abs(D * 3 - E) <= Matrix.fill(E.num_rows, E.num_cols, tol), "The matrix product was not computed successfully.")
        self.assertTrue(abs(F * G - H) <= Matrix.fill(H.num_rows, H.num_cols, tol), "The matrix product was not computed successfully.")
        with self.assertRaises(ValueError):
            G * F

    def test_transpose(self):
        A = Matrix([[1,2,3],[4,5,6]])
        B = Matrix([[1,4],[2,5],[3,6]])
        self.assertTrue(A.transpose() == B, "The matrix transpose was not computed successfully.")

class Test_Vec3d(unittest.TestCase):

    def test_constructor(self):
        pass

    def test_add(self):
        A = Vec3d(1,2,3)
        B = Vec3d(4,5,6)
        C = Vec3d(5,7,9)
        self.assertTrue(A + B == C, "The vector sum was not computed successfully.")
        self.assertTrue(B + A == C, "The vector sum was not computed successfully.")
        self.assertTrue(A + 3 == B, "The vector sum was not computed successfully.")
        self.assertTrue(3 + A == B, "The vector sum was not computed successfully.")

    def test_mul_with_matrix(self):
        x = Vec3d(1,2,3)
        A = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        b = Vec3d(14,32,50)
        self.assertTrue(Matrix.identity(3) * x == x, "The matrix multiplication with the vector was not computed successfully.")
        self.assertTrue(A * x == b, "The matrix multiplication with the vector was not computed successfully.")

if __name__ == '__main__':
    unittest.main()
