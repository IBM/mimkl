#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mimkl/definitions.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <stdexcept>

int main(int argc, char **argv)
{
    try
    {

        auto console = spdlog::stdout_color_mt("console");

        Eigen::Matrix<double, 2, 3> X;
        Eigen::SparseMatrix<double> L(3, 3);
        double a = 0.01;
        double b = 0.01;
        Eigen::Matrix<double, 2, 2> K_reference;

        X << 1., 2., 3., 4., 5., 6.;
        mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);

        K_reference << std::tanh(0.15), std::tanh(0.33), std::tanh(0.33),
        std::tanh(.78);
        Eigen::Matrix<double, 2, 2> K =
        mimkl::induction::induce_sigmoidal_kernel<MATRIX(double)>(X, X, L, a, b);

        assert(((K - K_reference).norm() < 0.0000000001) && "Identity Inducer");

        Eigen::SparseMatrix<double> L1(3, 3);
        typedef Eigen::Triplet<double> TripletDouble; // (row,col,coef)
        std::vector<TripletDouble> triplet_list;
        triplet_list.reserve(4);
        triplet_list.push_back(TripletDouble(0, 1, 1.));
        triplet_list.push_back(TripletDouble(1, 2, 1.));
        triplet_list.push_back(TripletDouble(1, 0, 1.));
        triplet_list.push_back(TripletDouble(2, 1, 1.));
        L1.setFromTriplets(triplet_list.begin(), triplet_list.end());

        K_reference << std::tanh(0.17), std::tanh(0.41), std::tanh(0.41),
        std::tanh(1.01);
        ;

        K = mimkl::induction::induce_sigmoidal_kernel<MATRIX(double)>(X, X, L1,
                                                                      a, b);

        std::cout << K << std::endl;
        std::cout << K_reference << std::endl;
        std::cout << (K - K_reference).norm() << std::endl;
        assert(((K - K_reference).norm() < 0.0000000001) &&
               "unweighted Graph Inducer");

        MATRIX(double) random_mat1 = MATRIX(double)::Random(2, 3);
        MATRIX(double) random_mat2 = MATRIX(double)::Random(4, 3);
        MATRIX(double) random_kernel;

        random_kernel =
        mimkl::induction::induce_sigmoidal_kernel<MATRIX(double)>(random_mat1,
                                                                  random_mat2,
                                                                  L1, a, b);
        console->info(" a random sigmoidal kernel\n{}", random_kernel);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
