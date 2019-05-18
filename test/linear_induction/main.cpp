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
        Eigen::Matrix<double, 2, 2> K_reference;

        X << 1., 2., 3., 4., 5., 6.;
        L.insert(0, 0) = 1.;
        K_reference << 1., 4., 4., 16.;

        Eigen::Matrix<double, 2, 2> K =
        mimkl::induction::induce_linear_kernel<MATRIX(double)>(X, X, L);
        assert(((K - K_reference).norm() == 0.0) &&
               "filter first gene Inducer");

        Eigen::SparseMatrix<double> L2(3, 3);
        mimkl::linear_algebra::fill_sparse_diagonal(L2, 1.0);
        K_reference << 14., 32., 32., 77.;

        K = mimkl::induction::induce_linear_kernel<MATRIX(double)>(X, X, L2);
        assert(((K - K_reference).norm() == 0.0) && "Identity Inducer");

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
