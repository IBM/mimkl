#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/io.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/average_mkl.hpp>
#include <spdlog/spdlog.h>
//#include <mimkl/models.hpp>
#include <mimkl/kernels_handler.hpp>
#include <mimkl/utilities.hpp>
#include <numeric>
#include <spdlog/fmt/ostr.h>

using dlib::mat;
using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;
using mimkl::utilities::check_invocable;
using mimkl::utilities::print_type;

using mimkl::kernels_handler::KernelsHandler;
using mimkl::models::AverageMKL;

//#define SPDLOG_DEBUG_ON
//#define SPDLOG_TRACE_ON

// Compile time log levels
// define SPDLOG_DEBUG_ON or SPDLOG_TRACE_ON
// SPDLOG_TRACE(console, "Enabled only #ifdef SPDLOG_TRACE_ON..{} ,{}",
// 1, 3.23);  SPDLOG_DEBUG(console, "Enabled only #ifdef SPDLOG_DEBUG_ON.. {}
// ,{}", 1, 3.23);

typedef std::function<MATRIX(double)(const MATRIX(double) &, const MATRIX(double) &)>
InducedFunction;
typedef std::vector<InducedFunction> Function_vec;
typedef std::vector<MATRIX(double)> Matrix_vec;

MATRIX(double)
test_average(const Function_vec function_vec,
             const Matrix_vec matrix_vec,
             const Matrix_vec matrix_vec_test,
             const bool precompute,
             const bool trace_normalization,
             std::shared_ptr<spdlog::logger> console,
             MATRIX(double) X,
             MATRIX(double) Y)
{
    console->info("\nRun with precompute = {} an trace_normalization = {}\n\n",
                  precompute, trace_normalization);

    console->info("rows X,Y : {},{}", X.rows(), Y.rows());
    console->info("sizes matric_vec : {},{}, sizes matric_vec_test : {},{}",
                  matrix_vec[0].rows(), matrix_vec[0].cols(),
                  matrix_vec_test[0].rows(), matrix_vec_test[0].cols());

    console->info("\nMatricial construction\n");

    AverageMKL<double, InducedFunction> average_mat =
    AverageMKL<double, InducedFunction>(matrix_vec, precompute,
                                        trace_normalization);
    console->info("constructor of average_mat done");

    average_mat.fit(matrix_vec);
    console->info("a prediction {}", average_mat.predict(matrix_vec_test));
    MATRIX(double) weight_mat_1 = average_mat.get_weights();
    try
    {
        average_mat.fit(matrix_vec_test);
        MATRIX(double) weight_mat_2 = average_mat.get_weights();
        console->error("not caught error on matricial non-squared.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on matricial non-squared. message: \n{}",
                      e.what());
    };
    try
    {
        average_mat.fit(X);
        MATRIX(double) weight_mat_fun = average_mat.get_weights();
        console->error("not caught error on matricial set_lhs.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on matricial set_lhs. message: \n{}",
                      e.what());
    };
    // retrain
    average_mat.fit(matrix_vec);

    console->info("\nFunctional construction\n");
    AverageMKL<double, InducedFunction> average(function_vec, precompute,
                                                trace_normalization);
    console->info("constructor of average_fun done");

    average.fit(matrix_vec);
    MATRIX(double) weight_fun_mat = average.get_weights();
    average.fit(X);
    MATRIX(double) weight_fun_1 = average.get_weights();
    // retrain
    average.fit(Y);
    MATRIX(double) weight_fun_2 = average.get_weights();
    try
    {
        average_mat.fit(matrix_vec_test);
        MATRIX(double) weight_fun_mat_2 = average_mat.get_weights();
        console->error(
        "not caught error on functional with non-squared matricial.");
    }
    catch (const std::exception &e)
    {
        console->warn("caught error on functional with non-squared matricial. "
                      "message: \n{}",
                      e.what());
    };

    assert(((weight_mat_1 - weight_fun_mat).norm() == 0) && "weight_mat_1");
    assert(((weight_fun_1 - weight_fun_mat).norm() == 0) && "weight_fun_1");

    return weight_fun_mat;
}

int main(int argc, char **argv)
{

    // Runtime log levels
    spdlog::set_level(spdlog::level::debug); // Set global log level to info
    auto console = spdlog::stdout_color_mt("console");

    const Index rows = 6; // 3 to reproduce single member in class  error
    const Index dims = 2;

    //
    //  ////
    //  MATRIX(double) X; //(rows, 2)

    //  std::string path = "../../data/simple_csv2.csv";
    //  console->info("path: {}",path);
    //  X = mimkl::io::eigen_matrix_from_csv<MATRIX(double)>(path, ','); //
    //  weird
    //  bug maybe here: floating point error that doesn't throw??
    //  ///////

    MATRIX(double) X(rows, dims);
    //  //	X << 1., 1., 3., 1., 1., 2.;
    //  X << 1., 1., 3., 1., 1., 4., 3., 2.;
    X << 1., 1., 3., 1., 1., 4., 3., 2., 1., -1., 3., -1.;
    //  ///////
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
        //	 spdlog::get("lambda")->debug("before actual inducer function
        // call" 			 "lhs rows {}  cols {} \n"
        //			 "rhs rows {}  cols {} \n"
        //			 "kernel rows {}  cols {} \n"
        //			 "inducer rows {}  cols {} \n",
        //			 lhs.rows(), lhs.cols(),
        //			 rhs.rows(), rhs.cols(),
        //			 kernel_matrix.rows(), kernel_matrix.cols(),
        //			 inducer.rows(), inducer.cols()
        //			 );
        return mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(
        lhs, rhs, inducer, degree, offset);
    };

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
    //
    //  MATRIX(double) K2(rows, rows);
    //  function_vec[0](X, X, K2);
    //  console->info("a kernel from function:\n{}", K2);
    //
    //  MATRIX(double) K3(rows, rows);
    //  function_vec[1](X, X, K3);
    //  console->info("another kernel from function:\n{}", K3);

    //  COLUMN(bool) all_true_I_swear = COLUMN(bool)::Constant(rows, true);
    //  console->info("bool:\n{}",all_true_I_swear);
    //  COLUMN(double) double_trouble = all_true_I_swear.cast<double>();
    //  console->info("double:\n{}",double_trouble);
    //
    //  console->info("mult by
    //  bool:\n{}",all_true_I_swear.cast<double>().array()
    //  * X.col(0).array());
    ////  console->info("mult by bool:\n{}",all_true_I_swear.cast<int>().array()
    ///*
    /// X.col(0).array());

    //////////
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
    std::vector<MATRIX(double)> matrix_vec_test(2);

    matrix_vec_test[0] = function_vec[0](X, Y);
    matrix_vec_test[1] = function_vec[1](X, Y);

    //////////
    //  AverageMKL<double, InducedFunction> average_mat(matrix_vec,true, true);
    //  AverageMKL<double, InducedFunction> average_mfun(function_vec,true, true);

    MATRIX(double)
    tt = test_average(function_vec, matrix_vec, matrix_vec_test, true, true,
                      console, X, Y);
    MATRIX(double)
    tf = test_average(function_vec, matrix_vec, matrix_vec_test, true, false,
                      console, X, Y);
    MATRIX(double)
    ft = test_average(function_vec, matrix_vec, matrix_vec_test, false, true,
                      console, X, Y);
    MATRIX(double)
    ff = test_average(function_vec, matrix_vec, matrix_vec_test, false, false,
                      console, X, Y);

    assert(((tt - ft).norm() == 0) && "trace_normalization true");
    assert(((ff - tf).norm() == 0) && "trace_normalization false");

    console->info("trace_normalization true: \n{}", tt);
    console->info("trace_normalization false: \n{}", tf);
    return EXIT_SUCCESS;
}
