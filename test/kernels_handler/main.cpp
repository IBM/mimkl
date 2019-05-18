#include "spdlog/spdlog.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <memory>
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
#include <spdlog/fmt/ostr.h>
#include <utility>

using dlib::mat;
using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;
using mimkl::kernels_handler::KernelsHandler;
using mimkl::utilities::check_invocable;
using mimkl::utilities::print_type;

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

void test_kernels_handler(const Function_vec function_vec,
                          const Matrix_vec matrix_vec,
                          const Matrix_vec matrix_vec_test,
                          const bool precompute,
                          const bool trace_normalization,
                          std::shared_ptr<spdlog::logger> console,
                          MATRIX(double) X,
                          MATRIX(double) Y,
                          COLUMN(double) w)
{
    console->trace("rows X,Y : {},{}", X.rows(), Y.rows());
    console->trace("sizes matric_vec : {},{}, sizes matric_vec_test : {},{}",
                   matrix_vec[0].rows(), matrix_vec[0].cols(),
                   matrix_vec_test[0].rows(), matrix_vec_test[0].cols());

    console->info("\n\n construction\n\n");
    MATRIX(double) outM, outF;

    KernelsHandler<double, InducedFunction> KHF(function_vec, precompute,
                                                trace_normalization);
    //	KHF.kernels->make_X_train_available(X);
    KHF.set_lhs(X);
    console->trace("KHF (train) sizes lhs,rhs : {},{}", KHF.get_lhs_size(),
                   KHF.get_rhs_size());
    KernelsHandler<double, InducedFunction> KHM(matrix_vec, precompute,
                                                trace_normalization);
    console->trace("KHM (train) sizes lhs,rhs : {},{}", KHM.get_lhs_size(),
                   KHF.get_rhs_size());

    console->info("\n\ntrain\n\n");

    console->info("[]");
    //	outF = KHF.kernels->operator [](0);
    //	outM = KHM.kernels->operator [](0);
    outF = KHF[0];
    outM = KHM[0];
    console->info("Functional access a kernel :\n{}", outF);
    console->info("Matricial access a kernel :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "[]");

    console->info("sum()");
    //	outF = KHF.kernels->sum();
    //	outM = KHM.kernels->sum();
    outF = KHF.sum();
    outM = KHM.sum();
    console->info("Functional sum() :\n{}", outF);
    console->info("Matricial sum() :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "sum()");

    console->info("weighted_sum()");
    //	outF = KHF.kernels->sum(w);
    //	outM = KHM.kernels->sum(w);
    outF = KHF.sum(w);
    outM = KHM.sum(w);
    console->info("Functional sum(weighted) :\n{}", outF);
    console->info("Matricial sum(weighted) :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "weighted_sum()");

    console->info("\n\ntest\n\n");
    KHF.set_rhs(Y);
    KHM.set_matrices(matrix_vec_test);
    console->trace("KHF (train) sizes lhs,rhs : {},{}", KHF.get_lhs_size(),
                   KHF.get_rhs_size());
    console->trace("KHM (train) sizes lhs,rhs : {},{}", KHM.get_lhs_size(),
                   KHF.get_rhs_size());

    console->info("[]");
    //	outF = KHF.kernels->operator [](0);
    //	outM = KHM.kernels->operator [](0);
    outF = KHF[0];
    outM = KHM[0];
    console->info("Functional access a kernel :\n{}", outF);
    console->info("Matricial access a kernel :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "[]");

    console->info("sum()");
    //	outF = KHF.kernels->sum();
    //	outM = KHM.kernels->sum();
    outF = KHF.sum();
    outM = KHM.sum();
    console->info("Functional sum() :\n{}", outF);
    console->info("Matricial sum() :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "sum()");

    console->info("weighted_sum()");
    //	outF = KHF.kernels->sum(w);
    //	outM = KHM.kernels->sum(w);
    outF = KHF.sum(w);
    outM = KHM.sum(w);
    console->info("Functional sum(weighted) :\n{}", outF);
    console->info("Matricial sum(weighted) :\n{}", outM);
    assert(((outF - outM).norm() == 0) && "weighted_sum()");
}

int main(int argc, char **argv)
{

    // Runtime log levels
    spdlog::set_level(spdlog::level::trace); // Set global log level to info
    auto console = spdlog::stdout_color_mt("console");

    const Index rows = 4; // 3 to reproduce single member in class  error
    const Index dims = 2;

    MATRIX(double) X(rows, 2);
    //	X << 1., 1., 3., 1., 1., 2.;
    X << 1., 1., 3., 1., 1., 4., 3., 2.;
    console->info("X\n{}", X);

    std::vector<std::string> labels;
    labels.reserve(rows);
    labels.push_back("a");
    labels.push_back("b");
    labels.push_back("a");
    labels.push_back("b");

    Eigen::SparseMatrix<double> L(dims, dims);
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

    MATRIX(double)
    K_ref0 = mimkl::induction::induce_linear_kernel<MATRIX(double)>(X, X, L);
    console->info("a kernel:\n{}", K_ref0);
    //
    MATRIX(double)
    K_ref1 =
    mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(X, X, L1, 1, 0);
    console->info("another kernel:\n{}", K_ref1);

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

    InducerFunction k_poly =
    [degree, offset](const MATRIX(double) & lhs, const MATRIX(double) & rhs,
                     const Eigen::SparseMatrix<double> inducer) {
        return mimkl::induction::induce_polynomial_kernel<MATRIX(double)>(
        lhs, rhs, inducer, degree, offset);
    };

    /////// here we go, preparing functional and matricial vectors
    std::vector<InducedFunction> function_vec =
    mimkl::induction::inducer_combination(k_poly, inducer_vec);

    MATRIX(double) K2 = function_vec[0](X, X);
    console->info("a kernel from function:\n{}", K2);

    MATRIX(double) K3 = function_vec[1](X, X);
    console->info("another kernel from function:\n{}", K3);

    std::vector<MATRIX(double)> matrix_vec(2);
    matrix_vec[0] = K2;
    matrix_vec[1] = K3;

    assert(((K_ref0 - K2).norm() == 0) && "Identity Inducer");
    assert(((K_ref1 - K3).norm() == 0) && "non-identity Inducer");

    console->info("Matricial vector invocable : {}",
                  check_invocable(matrix_vec[0]));
    console->info("Functional vector invocable : {}",
                  check_invocable(function_vec[0]));

    MATRIX(double) Y(6, 2);
    Y << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.;
    console->info("Y\n{}", Y);
    std::vector<MATRIX(double)> matrix_vec_test(2);

    matrix_vec_test[0] = function_vec[0](X, Y);
    matrix_vec_test[1] = function_vec[1](X, Y);

    console->trace("some test kernel:\n{}", matrix_vec_test[0]);
    ////////
    MATRIX(double) *X_ptr = &X;
    MATRIX(double) *Y_ptr = &Y;
    console->trace("some test kernel from dereferenced pointers :\n{}",
                   function_vec[0](*X_ptr, *Y_ptr));

    ////////
    COLUMN(double) w(2);
    w << 1., 1.;
    MATRIX(double) out;

    //
    console->info("precomputed=false trace_normalization=false");
    test_kernels_handler(function_vec, matrix_vec, matrix_vec_test, false,
                         false, console, X, Y, w);

    console->info("precomputed=true trace_normalization=false");
    test_kernels_handler(function_vec, matrix_vec, matrix_vec_test, true, false,
                         console, X, Y, w);

    console->info("precomputed=false trace_normalization=true");
    test_kernels_handler(function_vec, matrix_vec, matrix_vec_test, false, true,
                         console, X, Y, w);

    console->info("default precomputed=true trace_normalization=true");
    test_kernels_handler(function_vec, matrix_vec, matrix_vec_test, true, true,
                         console, X, Y, w);
}
