#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/io.hpp>
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

        Eigen::Matrix<double, 5, 3> X_ref;
        X_ref << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.;
        std::cout << "X_ref\n" << X_ref << std::endl;

        // csv
        Eigen::MatrixXd X;
        std::cout << "X declared\n" << X << std::endl;

        std::string path = "../../data/simple_csv.csv";
        std::cout << "a full path\n" << path << std::endl;

        //// relative to mimkl-build/ (eg. eclipse)
        //  const std::string path =
        //  std::string("../mimkl/test/data/simple_csv.csv");
        //  std::cout <<"path\n" << path << std::endl;

        std::cout << "return value\n"
                  << mimkl::io::eigen_matrix_from_csv<Eigen::MatrixXd>(path, ',')
                  << std::endl;

        X = mimkl::io::eigen_matrix_from_csv<Eigen::MatrixXd>(path);
        std::cout << "X\n" << X << std::endl;

        assert(((X - X_ref).norm() == 0) && "read a simple csv to matrix");

        // tab delimited
        path = "../../data/simple_tsv.txt";
        std::cout << "a full path\n" << path << std::endl;

        Eigen::MatrixXd Tab;
        Tab = mimkl::io::eigen_matrix_from_csv<Eigen::MatrixXd>(path, '\t');
        std::cout << "Tab\n" << Tab << std::endl;

        assert(((Tab - X_ref).norm() == 0) && "read a simple csv to matrix");

        // CSVReader
        // none
        mimkl::io::HeaderPolicy header_policy = mimkl::io::none;

        DataFrame dfTab =
        mimkl::io::CSVReader::read<Eigen::MatrixXd>(path, '\t', header_policy);
        std::cout << "dfTab\n" << dfTab << std::endl;
        assert(((dfTab.matrix() - X_ref).norm() == 0) &&
               "read a simple tsv to dataframe");

        // both
        header_policy = mimkl::io::both;
        path = "../../data/both_headers_tsv.txt";
        DataFrame dfboth =
        mimkl::io::CSVReader::read<Eigen::MatrixXd>(path, '\t', header_policy);
        std::cout << "dfboth\n" << dfboth << std::endl;
        assert(((dfboth.matrix() - X_ref).norm() == 0) &&
               "read a simple tsv to dataframe");

        // index_only
        header_policy = mimkl::io::index_only;
        path = "../../data/index_only_headers_tsv.txt";
        DataFrame dfindex_only =
        mimkl::io::CSVReader::read<Eigen::MatrixXd>(path, '\t', header_policy);
        std::cout << "dfindex_only\n" << dfindex_only << std::endl;
        assert(((dfindex_only.matrix() - X_ref).norm() == 0) &&
               "read a simple tsv to dataframe");

        // column_only
        header_policy = mimkl::io::column_only;
        path = "../../data/column_only_headers_tsv.txt";
        DataFrame dfcolumn_only =
        mimkl::io::CSVReader::read<Eigen::MatrixXd>(path, '\t', header_policy);
        std::cout << "dfcolumn_only\n" << dfcolumn_only << std::endl;
        assert(((dfcolumn_only.matrix() - X_ref).norm() == 0) &&
               "read a simple tsv to dataframe");
        return EXIT_SUCCESS;
    }

    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}
