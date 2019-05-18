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

        double s_squared = 1.;
        Eigen::Matrix<double, 2, 2> K_reference;

        X << 1., 2., 3., 4., 5., 6.;
        mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);

        K_reference << std::exp(-0.), std::exp(-27. / (2. * s_squared)),
        std::exp(-27. / (2. * s_squared)), std::exp(-0.);
        Eigen::Matrix<double, 2, 2> K =
        mimkl::induction::induce_gaussian_kernel<Eigen::Matrix<double, 2, 2>>(
        X, X, L, s_squared);
        assert(((K - K_reference).norm() == 0.0) && "Identity Inducer");

        Eigen::SparseMatrix<double> L1(3, 3);
        typedef Eigen::Triplet<double> TripletDouble; // (row,col,coef)
        std::vector<TripletDouble> triplet_list;
        triplet_list.reserve(4);
        triplet_list.push_back(TripletDouble(0, 1, 1.));
        triplet_list.push_back(TripletDouble(1, 2, 1.));
        triplet_list.push_back(TripletDouble(1, 0, 1.));
        triplet_list.push_back(TripletDouble(2, 1, 1.));
        L1.setFromTriplets(triplet_list.begin(), triplet_list.end());

        K_reference << std::exp(-0. / (2. * s_squared)),
        std::exp(-36. / (2. * s_squared)), std::exp(-36. / (2. * s_squared)),
        std::exp(-0.);

        K = mimkl::induction::induce_gaussian_kernel<Eigen::Matrix<double, 2, 2>>(
        X, X, L1, s_squared);
        assert(((K - K_reference).norm() == 0.0) && "unweighted Graph Inducer");

        MATRIX(double) random_mat1 = MATRIX(double)::Random(2, 3);
        MATRIX(double) random_mat2 = MATRIX(double)::Random(4, 3);
        MATRIX(double) random_kernel;

        //    random_kernel =
        //    mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(random_mat1,
        //    random_mat2, L1, 2.,1.);
        //    console->info(" a random pol kernel\n{}", random_kernel);

        random_kernel =
        mimkl::induction::induce_gaussian_kernel<MATRIX(double)>(random_mat1,
                                                                 random_mat2,
                                                                 L1, s_squared);
        console->info(" a random gaussian kernel\n{}", random_kernel);

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
