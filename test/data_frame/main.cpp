#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <stdexcept>

using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;

int main(int argc, char **argv)
{
    try
    {
        Eigen::Matrix<double, 2, 3> X;
        X << 1., 2., 3., 4., 5., 6.;

        Eigen::MatrixXd Y(4, 3);
        Y << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.;

        // check range()
        std::vector<int> vec = range(X.cols()); //
        std::cout << "a range:\n";
        for (std::vector<int>::const_iterator i = vec.begin(); i != vec.end();
             ++i)
            std::cout << *i << " ";
        std::cout << std::endl;
        // check indexing_from_vector()
        Indexing an_indexing = indexing_from_vector(vec);
        std::cout << "first element of an Indexing\n"
                  << an_indexing.begin()->second << std::endl;

        // test constructors
        DataFrame df;       // default
        DataFrame df1 = df; // trivial
        DataFrame dfY = Y;  // assignment
        df = Y;
        DataFrame dfX = X;

        DataFrame df_copy(X);              // copy
        std::cout << df_copy << std::endl; // operator<<

        DataFrame df_copy_withindex(X, indexing_from_vector(range(X.rows())),
                                    indexing_from_vector(range(X.cols()))); // copy

        // construct from given vector of strings
        std::vector<std::string> labels;
        labels.push_back("patient_1");
        labels.push_back("patient_2");
        labels.push_back("patient_3");
        labels.push_back("patient_3");
        Indexing another_indexing;
        Index i = 0;
        for (const auto &element : labels)
        {
            another_indexing.emplace(element, i++);
        }
        std::cout << "first key of an Indexing\n"
                  << another_indexing.begin()->first << std::endl;
        DataFrame df_ylabels(Y, another_indexing, an_indexing); // copy
        std::cout << "duplicate row labels df\n" << df_ylabels << std::endl;

        //// slicing
        // no such index found
        try
        {
            df_copy["no_way_josé"];
        }
        catch (...)
        {
            std::cout << "it's ok josé" << std::endl;
        }

        DataFrame col_slice = df_copy["1"]; // 2nd col
        std::cout << "col slice\n" << col_slice << std::endl;
        DataFrame row_slice = df_copy.loc("1"); // 2nd row

        Eigen::MatrixXd row_mat(1, 3);
        row_mat << 4., 5., 6.;
        DataFrame df_row_mat(row_mat);
        row_slice = row_slice - df_row_mat; // some df operation
        std::cout << "row slice with substraction\n" << row_slice << std::endl;
        std::cout << ".norm()\n" << row_slice.norm() << std::endl;

        std::cout << df_ylabels << std::endl;
        DataFrame labeled_row = df_ylabels.loc("patient_2"); // 2nd row
        std::cout << "a labeled row:\n" << labeled_row << std::endl;
        // duplicate label
        DataFrame duplicate_row = df_ylabels.loc("patient_3"); // 2nd row
        std::cout << "two rows with same label:\n"
                  << duplicate_row << std::endl;

        // chained
        DataFrame an_element = (df_ylabels.loc("patient_2"))["0"]; // row and col
        std::cout << "chained, an element:\n" << an_element << std::endl;

        std::vector<std::string> col_vec(2);
        col_vec.push_back("0");
        col_vec.push_back("2");
        DataFrame two_element =
        df_ylabels.loc("patient_2")[col_vec]; // row and col
        std::cout << "chained, a row two elements:\n"
                  << two_element << std::endl;

        Eigen::MatrixXd row_ref(1, 2);
        row_ref << 4., 6.;
        std::cout << "matrix row :\n" << row_ref << std::endl;

        DataFrame ref = row_ref;
        std::cout << "df row :\n" << ref << std::endl;

        Eigen::MatrixXd m;
        m = ref.matrix(); // protected constructor
        std::cout << "get plain matrix back :\n"
                  << m << std::endl; // no indexing
        assert((ref.norm() == row_ref.norm()) && " norm on df ");

        assert(((two_element - row_ref).norm() == 0.0) &&
               "matrix operation on DF and matrix");
        assert(((two_element - ref).norm() == 0.0) &&
               "matrix operation on DFs");

        // multiplication, lin. kernel
        Eigen::Matrix<double, 2, 2> K_lin_ref;
        K_lin_ref << 1., 4., 4., 16.;
        Eigen::SparseMatrix<double> L(3, 3);
        L.insert(0, 0) = 1.;
        assert(
        (((dfX * L.selfadjointView<Eigen::Lower>() * dfX.adjoint()) - K_lin_ref)
         .norm() == 0.0) &&
        "df multiplication / linear kernel directly");

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
