import unittest

from integrationutils.util.base_objects import Vec3d, Matrix3d

class Test_Vec3d(unittest.TestCase):

    def test_constructor(self):
        pass

class Test_Matrix3d(unittest.TestCase):

    def test_constructor(self):
        pass

    def test_fill(self):
        print(Matrix3d.fill(1.0))

if __name__ == '__main__':
    unittest.main()
