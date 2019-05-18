import unittest
from functools import partial

import numpy as np
import pymimkl as pm
"""if the library is not installed, make sure the .so/.dlib file
is either in the working directory or on the PYTHONPATH
`PYTHONPATH=/path/to/pymimkl.version.so python test.py`
"""


class pymimklTestCase(unittest.TestCase):
    def test_easymkl(self):
        def f1(lhs, rhs):
            return lhs.dot(rhs.T)

        f2 = partial(pm.polynomial_kernel, degree=2, offset=0)
        f3 = partial(pm.polynomial_kernel, degree=3, offset=0)

        funs = [f1, f2, f3]

        X = np.array(
            [[1,  1], [3,  1], [1,  4], [3,  2], [1, -1], [3, -1]],
            dtype=np.float64, order='F'
        )
        labels = ['a', 'b', 'a', 'b', 'c', 'c']

        print(X)
        easy = pm.EasyMKL(funs)
        easy.lam = 0.5
        easy.fit(X, labels)
        print(easy.etas)
        np.testing.assert_almost_equal(
            easy.etas.sum(axis=0),
            [1.0, 1.0, 1.0],
            decimal=16  # np.finfo(float).eps
        )

# if __name__ == '__main__':
#     unittest.main()
