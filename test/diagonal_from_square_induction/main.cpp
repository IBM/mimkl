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
        Eigen::Matrix<double, 2, 3> X;
        Eigen::SparseMatrix<double> L(3, 3);
        Eigen::Matrix<double, 2, 1> diag;
        Eigen::Matrix<double, 2, 1> diag_reference;

        X << 1., 2., 3., 4., 5., 6.;
        mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);

        diag_reference << 14., 77.;

        mimkl::induction::get_diagonal_from_square_induction(X, L, diag);
        assert(((diag - diag_reference).norm() == 0.0) && "Identity Inducer");

        Eigen::SparseMatrix<double> L1(3, 3);
        typedef Eigen::Triplet<double> TripletDouble; // (row,col,coef)
        std::vector<TripletDouble> triplet_list;
        triplet_list.reserve(4);
        triplet_list.push_back(TripletDouble(0, 1, 1.));
        triplet_list.push_back(TripletDouble(1, 2, 1.));
        triplet_list.push_back(TripletDouble(1, 0, 1.));
        triplet_list.push_back(TripletDouble(2, 1, 1.));
        L1.setFromTriplets(triplet_list.begin(), triplet_list.end());

        diag_reference << 16., 100.;

        mimkl::induction::get_diagonal_from_square_induction(X, L1, diag);
        assert(((diag - diag_reference).norm() == 0.0) &&
               "unweighted Graph Inducer");

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
