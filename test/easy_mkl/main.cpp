#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/io.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/kernels_handler.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/easy_mkl.hpp>
#include <mimkl/solvers/komd.hpp>
#include <mimkl/utilities.hpp>
#include <numeric>
#include <ostream>
#include <spdlog/fmt/ostr.h>
//#include <Eigen/Cholesky> // check if inducer is sdp llt() ...

using dlib::mat;
using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;
using mimkl::kernels_handler::KernelsHandler;
using mimkl::models::EasyMKL;
using mimkl::utilities::check_invocable;
using mimkl::utilities::print_type;
#define SPDLOG_DEBUG_ON
#define SPDLOG_TRACE_ON

// Compile time log levels
// define SPDLOG_DEBUG_ON or SPDLOG_TRACE_ON
// SPDLOG_TRACE(console, "Enabled only #ifdef SPDLOG_TRACE_ON..{} ,{}",
// 1, 3.23);  SPDLOG_DEBUG(console, "Enabled only #ifdef SPDLOG_DEBUG_ON.. {}
// ,{}", 1, 3.23);

typedef std::function<MATRIX(double)(const MATRIX(double) &, const MATRIX(double) &)>
InducedFunction;
typedef std::vector<InducedFunction> Function_vec;
typedef std::vector<MATRIX(double)> Matrix_vec;

// snipped from http://en.cppreference.com/w/cpp/container/vector/vector
template <typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &v)
{
    s.put('[');
    char comma[3] = {'\0', ' ', '\0'};
    for (const auto &e : v)
    {
        s << comma << e;
        comma[0] = ',';
    }
    return s << ']';
}

MATRIX(double)
test_easy_mkl(const Function_vec function_vec,
              const Matrix_vec matrix_vec,
              const Matrix_vec matrix_vec_test,
              const bool precompute,
              const bool trace_normalization,
              std::shared_ptr<spdlog::logger> console,
              MATRIX(double) X,
              MATRIX(double) Y,
              std::vector<std::string> labels,
              std::vector<std::string> labels_y)
{
    console->info("\n\nRun with precompute = {} and trace_normalization = "
                  "{}\n\n",
                  precompute, trace_normalization);

    console->info("rows X,Y : {},{}", X.rows(), Y.rows());
    console->info("sizes matric_vec : {},{}, sizes matric_vec_test : {},{}",
                  matrix_vec[0].rows(), matrix_vec[0].cols(),
                  matrix_vec_test[0].rows(), matrix_vec_test[0].cols());

    console->info("\nMatricial construction\n");

    EasyMKL<double, InducedFunction> easy_mat =
    EasyMKL<double, InducedFunction>(matrix_vec, precompute, trace_normalization);
    console->info("constructor of easy_mat done");

    easy_mat.set_lambda(0.8);

    easy_mat.fit(matrix_vec, labels);
    MATRIX(double) beta_mat_1 = easy_mat.get_etas();
    MATRIX(double) gammas_mat_1 = easy_mat.get_gammas();
    MATRIX(double) biases_mat_1 = easy_mat.get_biases();
    std::vector<std::string> pred_mat_1 = easy_mat.predict(matrix_vec_test);
    try
    {
        easy_mat.fit(matrix_vec_test, labels);
        MATRIX(double) beta_mat_2 = easy_mat.get_etas();
        MATRIX(double) gammas_mat_2 = easy_mat.get_gammas();
        MATRIX(double) biases_mat_2 = easy_mat.get_biases();
        std::vector<std::string> pred_mat_2 = easy_mat.predict(matrix_vec_test);

        console->error("not caught error on matricial non-squared.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on matricial non-squared. message: \n{}",
                      e.what());
    };
    try
    {
        easy_mat.fit(X, labels);
        MATRIX(double) beta_mat_fun = easy_mat.get_etas();
        MATRIX(double) gammas_mat_fun = easy_mat.get_gammas();
        MATRIX(double) biases_mat_fun = easy_mat.get_biases();
        std::vector<std::string> pred_mat_fun = easy_mat.predict(Y);
        console->error("not caught error on matricial set_lhs.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on matricial set_lhs. message: \n{}",
                      e.what());
    };
    // retrain
    easy_mat.fit(matrix_vec, labels);

    console->info("\nFunctional construction\n");

    EasyMKL<double, InducedFunction> easy(function_vec, precompute,
                                          trace_normalization);
    console->info("constructor of easy_fun done");

    easy.set_lambda(0.8);

    easy.fit(matrix_vec, labels);
    MATRIX(double) beta_fun_mat = easy.get_etas();
    MATRIX(double) gammas_fun_mat = easy.get_gammas();
    MATRIX(double) biases_fun_mat = easy.get_biases();
    console->info("fitted");
    try
    {
        std::vector<std::string> pred_fun_mat_fun =
        easy.predict(Y); // lhs unavailable ...
        console->error("not caught error on functional missing set_lhs.");
    }
    catch (const std::exception &e)
    {
        console->warn(
        "caught error on functional missing set_lhs. message: \n{}", e.what());
    };
    std::vector<std::string> pred_fun_mat_mat = easy.predict(matrix_vec_test);

    //
    easy.fit(X, labels);
    MATRIX(double) beta_fun_1 = easy.get_etas();
    MATRIX(double) gammas_fun_1 = easy.get_gammas();
    MATRIX(double) biases_fun_1 = easy.get_biases();
    std::vector<std::string> pred_fun_1 = easy.predict(Y);
    std::vector<std::string> pred_fun_mat_1 = easy.predict(matrix_vec_test);

    console->info("retrain");

    // retrain

    easy.fit(Y, labels_y);
    MATRIX(double) beta_fun_2 = easy.get_etas();
    MATRIX(double) gammas_fun_2 = easy.get_gammas();
    MATRIX(double) biases_fun_2 = easy.get_biases();
    console->info("");

    std::vector<std::string> pred_fun_2 = easy.predict(Y);

    console->info("");
    try
    {
        std::vector<std::string> pred_fun_mat_2 =
        easy.predict(matrix_vec_test); // works only if trained on X
        console->error(
        "not caught error on functional prediction with mismatching "
        "kernel_size.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on functional prediction with mismatching "
                      "kernel_size. message: \n{}",
                      e.what());
    };
    console->info("");

    try
    {
        easy_mat.fit(matrix_vec_test, labels);
        MATRIX(double) beta_fun_mat_2 = easy.get_etas();
        MATRIX(double) gammas_fun_mat_2 = easy.get_gammas();
        MATRIX(double) biases_fun_mat_2 = easy.get_biases();
        console->error(
        "not caught error on functional with non-squared matricial.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on functional with non-squared matricial. "
                      "message: \n{}",
                      e.what());
    };

    assert(((beta_mat_1 - beta_fun_mat).norm() == 0) && "beta_mat_1");
    assert(((beta_fun_1 - beta_fun_mat).norm() == 0) && "beta_fun_1");
    assert(((gammas_mat_1 - gammas_fun_mat).norm() == 0) && "beta_mat_1");
    assert(((gammas_fun_1 - gammas_fun_mat).norm() == 0) && "beta_fun_1");
    assert(((biases_mat_1 - biases_fun_mat).norm() == 0) && "beta_mat_1");
    assert(((biases_fun_1 - biases_fun_mat).norm() == 0) && "beta_fun_1");

    std::cout << pred_mat_1 << std::endl;
    std::cout << pred_fun_mat_mat << std::endl;
    std::cout << pred_fun_1 << std::endl;
    ;
    std::cout << pred_fun_mat_1 << std::endl;

    assert((pred_mat_1 == pred_fun_mat_mat) && (pred_mat_1 == pred_fun_1) &&
           pred_mat_1 == pred_fun_mat_1);

    std::cout << pred_mat_1 << std::endl;
    std::cout << pred_fun_2 << std::endl;
    return beta_fun_mat;
}

int main(int argc, char **argv)
{

    // Runtime log levels
    spdlog::set_level(spdlog::level::info); // Set global log level to info
    auto console = spdlog::stdout_color_mt("console");
    spdlog::get("EasyMKL")->set_level(spdlog::level::debug);
    spdlog::get("KernelsHandler")->set_level(spdlog::level::debug);

    const Index rows = 6; // 3 to reproduce single member in class  error
    const Index dims = 2;

    MATRIX(double) X(rows, 2);
    //	X << 1., 1., 3., 1., 1., 2.;
    X << 1., 1., 3., 1., 1., 4., 3., 2., 1., -1., 3., -1.;
    console->info("X\n{}", X);

    std::vector<std::string> labels;
    labels.reserve(rows);
    labels.push_back("a");
    labels.push_back("b");
    labels.push_back("a");
    labels.push_back("b");
    labels.push_back("c");
    labels.push_back("c");

    Eigen::SparseMatrix<double> L(2, 2);
    mimkl::linear_algebra::fill_sparse_diagonal(L, 1.0);

    Eigen::SparseMatrix<double> L1(dims, dims);
    typedef Eigen::Triplet<double> TripletDouble; // (row,col,coef)
    std::vector<TripletDouble> triplet_list;
    triplet_list.reserve(4);
    triplet_list.push_back(TripletDouble(0, 1, 1.));
    //  triplet_list.push_back(TripletDouble(1, 2, 1.));
    triplet_list.push_back(TripletDouble(1, 0, 1.));
    //  triplet_list.push_back(TripletDouble(2, 1, 1.));
    L1.setFromTriplets(triplet_list.begin(), triplet_list.end());

    // L1: (not SDP)
    // 0 1
    // 1 0

    std::vector<Eigen::SparseMatrix<double>> inducer_vec;
    inducer_vec.reserve(2);
    inducer_vec.push_back(L);
    inducer_vec.push_back(L1);

    typedef std::function<MATRIX(double)(const MATRIX(double) &, const MATRIX(double) &,
                                         const Eigen::SparseMatrix<double>)>
    InducerFunction;
    typedef std::function<MATRIX(double)(const MATRIX(double) &,
                                         const MATRIX(double) &)>
    InducedFunction;

    const double degree = 1.;
    const double offset = 0.;

    auto lambda_logger = spdlog::stdout_color_mt("lambda");
    InducerFunction k_poly =
    [degree, offset](const MATRIX(double) & lhs, const MATRIX(double) & rhs,
                     const Eigen::SparseMatrix<double> inducer) {
        spdlog::get("lambda")->trace(
        "about to call inducer function\n"
        "inducer \n{}\n"
        //        			 "lhs rows {}  cols {} \n"
        //        			 "rhs rows {}  cols {} \n"
        //        			 "kernel rows {}  cols {} \n"
        //        			 "inducer rows {}  cols {} \n"
        "type of function\n{}",
        inducer,
        //        			 lhs.rows(), lhs.cols(),
        //        			 rhs.rows(), rhs.cols(),
        //        			 kernel_matrix.rows(),
        //        kernel_matrix.cols(),
        //        			 inducer.rows(), inducer.cols(),
        print_type(mimkl::induction::induce_polynomial_kernel<
                   MATRIX(double), MATRIX(double), MATRIX(double),
                   Eigen::SparseMatrix<double>>));
        return mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(
        lhs, rhs, inducer, degree, offset);
    };

    MATRIX(double) K(rows, rows);
    console->info("\e[1;32mtype: \e[0m \n{}",
                  print_type<decltype(k_poly(X, X, L))>());
    console->info("decltype argument evaluated?:\n{}", K);
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(k_poly));
    console->info("check_invocable? {}", check_invocable(k_poly));

    std::vector<InducedFunction> function_vec =
    mimkl::induction::inducer_combination(k_poly, inducer_vec);
    //// 	==inducer_combination:
    //	  for (Eigen::SparseMatrix<double> inducer : inducer_vec) {
    //		  function_vec.push_back(
    //			[&,inducer](const MATRIX(double) &lhs,
    //					  const MATRIX(double)&rhs,
    //					  const MATRIX(double) &kernel_matrix) {
    //			  k_poly(lhs, rhs, kernel_matrix, inducer);
    //			});
    //
    //	  }

    MATRIX(double) K2 = function_vec[0](X, X);
    console->info("a kernel from function:\n{}", K2);

    MATRIX(double) K3 = function_vec[1](X, X);
    console->info("another kernel from function:\n{}", K3);

    std::vector<MATRIX(double)> matrix_vec(2);
    matrix_vec[0] = K2;
    matrix_vec[1] = K3;

    MATRIX(double) Y(7, 2);
    Y << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.;
    console->info("Y\n{}", Y);
    std::vector<std::string> labels_y{"frogurt",
                                      "frogurt",
                                      "frogurt",
                                      "frogurt",
                                      "Pan-Galactic Gargle Blaster",
                                      "Pan-Galactic Gargle Blaster",
                                      "Pan-Galactic Gargle Blaster"};

    std::vector<MATRIX(double)> matrix_vec_test(2);

    matrix_vec_test[0] = function_vec[0](X, Y);
    matrix_vec_test[1] = function_vec[1](X, Y);

    MATRIX(double)
    ft = test_easy_mkl(function_vec, matrix_vec, matrix_vec_test, false, true,
                       console, X, Y, labels, labels_y);
    MATRIX(double)
    ff = test_easy_mkl(function_vec, matrix_vec, matrix_vec_test, false, false,
                       console, X, Y, labels, labels_y);
    MATRIX(double)
    tf = test_easy_mkl(function_vec, matrix_vec, matrix_vec_test, true, false,
                       console, X, Y, labels, labels_y);
    MATRIX(double)
    tt = test_easy_mkl(function_vec, matrix_vec, matrix_vec_test, true, true,
                       console, X, Y, labels, labels_y);

    assert(((tt - ft).norm() == 0) && "trace_normalization true");
    assert(((ff - tf).norm() == 0) && "trace_normalization false");

    console->info("trace_normalization true: \n{}", tt);
    console->info("trace_normalization false: \n{}", tf);

    // remember L1 is not SDP so weights can turn out negative
    return EXIT_SUCCESS;
}
