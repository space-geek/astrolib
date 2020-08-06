import unittest

from integrationutils.util.base_objects import Matrix
from integrationutils.util.base_objects import Vec3d

class Test_Matrix(unittest.TestCase):

    def test_constructor(self):
        pass

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

    # def test_subtract(self):
    #     A = Matrix([[1,2,3],[4,5,6]])
    #     B = Matrix([[4,5,6],[7,8,9]])
    #     self.assertTrue(A - B == Matrix.fill(3,2,-3), "The matrix difference was not computed successfully.")
    #     self.assertTrue(B - A == Matrix.fill(3,2,3), "The matrix difference was not computed successfully.")

class Test_Vec3d(unittest.TestCase):

    def test_constructor(self):
        pass

    def test_add(self):
        A = Vec3d(1,2,3)
        B = Vec3d(4,5,6)
        C = Vec3d(5,7,9)
        self.assertTrue(A + B == C, "The vector sum was not computed successfully.")

if __name__ == '__main__':
    unittest.main()
