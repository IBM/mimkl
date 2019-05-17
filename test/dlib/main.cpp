#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <dlib/optimization.h>
#include <iostream>
#include <limits>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/io.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/easy_mkl.hpp>
#include <mimkl/solvers/komd.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using dlib::mat;
using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::debug); // Set global log level to info
    auto console = spdlog::stdout_color_mt("console");

    const Index rows = 4; // 3 to reproduce error

    MATRIX(double) X(rows, 2);
    //	X << 1., 1., 3., 1., 1., 2.;
    X << 1., 1., 3., 1., 1., 4., 3., 2.;

    Eigen::SparseMatrix<double> L(2, 2);
    mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);
    MATRIX(double)
    K = mimkl::induction::induce_linear_kernel<MATRIX(double)>(X, X, L);
    COLUMN(double) Y(rows);
    //	Y << -1., 1., -1.;
    Y << -1., 1., -1., 1.;
    //	Eigen::Matrix<double, rows, 1> alphas;
    //	Eigen::Matrix<double, 3, 1> gamma_ref;

    std::cout << "X\n" << X << std::endl;
    std::cout << "dlib X\n" << mat(X) << std::endl;
    std::cout << "Y\n" << Y << std::endl;

    // test dlib assertions do not trigger? eg. with y.size()==1  TODO

    // KOMD
    mimkl::solvers::KOMD<double> some_dots(K, 0.2, 1e-8);
    some_dots.solve(Y);
    console->debug("get result :\n{}",
                   some_dots.get_result()); // gamma is private
    //  typedef Eigen::Map<EigenCol> MapEigenCol;
    COLUMN(double)
    gamma = some_dots.get_result(); // conversion of MapEigenCol to EigenCol
    console->info("gamma eigen (map to mat):\n{}", gamma);
    return EXIT_SUCCESS;
}
