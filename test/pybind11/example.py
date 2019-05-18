#!/usr/bin/env python

import example
import numpy as np
import scipy.sparse

t = example.Example()
t.set('bom dia!')
print(t.greet())

t.many(['Good Morning', 'Buon giorno', 'Kali mera'])
print(t.greet())

print(example.add_to_3(2))
print(example.apply_5(example.add_to_3))
print(example.apply_5(lambda x: x + 4))

xs = [1, 2, 3, 4, 5]
print(xs)
print(example.map1(example.add_to_3, xs))
print(example.map1(lambda x: x + 4, xs))
print(xs)
fs = [(lambda a: lambda x: x + a)(a) for a in xs]
print(example.zip_map(fs, xs))

m = np.array([[1, 2], [3, 4]])
print(m)
print(example.a_mat)
print(example.a_mat())


L = scipy.sparse.identity(2).tocsc()
print(example.add_sparse(L, L))
print(example.add_sparse(L, L).toarray())
L2 = scipy.sparse.csc_matrix(np.array([[0, 1], [1, 0]]))


print(np.array([[1, 2], [3, 4]]) + np.array([[1, 2], [3, 4]]))
print(example.sum_2_mats(
    np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])
))
ms = [m, m, m]
print(example.sum_plain_matrices(ms))  # non-template on vector of matrices

print(
    'templatized on vector of matrix (specialized from mimkl::linear_algebra)'
)
print('{}'.format(example.sum_plain_matrices_lin_alg(ms)))


def linear_kernel(lhs, rhs):
    return np.dot(lhs, rhs.T)


print('a linear kernel\n{}'.format(linear_kernel(m, m)))
print('a linear kernel from c++\n{}'.format(example.linear_kernel_cpp(m, m)))

mfs = [
    linear_kernel, (lambda lhs, rhs: np.dot(lhs, rhs.T)),
    example.linear_kernel_cpp
]
print('non-template on vector of function\n{}'.format(
    example.sum_matrices(m, m, mfs)
))

print('templatized on vector of function --testing--\n{}'.format(
    example.sum_matrices_temp(m, m, mfs)
))

print('templatized on vector of function but w/o std::accumulate --testing--')
print('{}'.format(example.sum_matrices_temp_inline(m, m, mfs)))

double_fs = [(lambda x, y: x + 1), (lambda x, y: x + y * 10)]
print('templatized on vector of function ( just double not Mat) --testing--')
print('{}'.format(example.sum_temp(1, 2, double_fs)))

mat_double_fs = [(lambda x, y: float(1)), (lambda x, y: float(10))]
print(
    'templatized on vector of function (returning double) --testing--'
)
print('{}'.format(example.sum_mat_return_scalar_temp(m, m, mat_double_fs)))

