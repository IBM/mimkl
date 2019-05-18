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
        int size = 3;
        Eigen::SparseMatrix<double> L(size, size);
        Eigen::SparseMatrix<double> L_reference(size, size);

        // L
        mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);
        // L_reference
        for (int i = 0; i < size; i++)
        {
            L_reference.insert(i, i) = 1.;
        }

        assert((L - L_reference).norm() == 0.0);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
