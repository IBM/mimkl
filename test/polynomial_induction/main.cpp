#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <stdexcept>

int main(int argc, char **argv)
{
    try
    {
        Eigen::Matrix<double, 2, 3> X;
        Eigen::SparseMatrix<double> L(3, 3);
        double p;
        double c;
        Eigen::Matrix<double, 2, 2> K_reference;

        X << 1., 2., 3., 4., 5., 6.;
        mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);
        p = 1;
        c = 0;
        K_reference << 14., 32., 32., 77.;

        Eigen::Matrix<double, 2, 2> K =
        mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(X, X, L, p, c);
        assert((K - K_reference).norm() == 0.0);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
