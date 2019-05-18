#include "main.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/io.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/easy_mkl.hpp>
#include <mimkl/solvers/komd.hpp>
#include <mimkl/utilities.hpp>
#include <numeric>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using dlib::mat;
using mimkl::data_structures::DataFrame;
using mimkl::data_structures::indexing_from_vector;
using mimkl::data_structures::range;
using mimkl::definitions::Indexing;
using mimkl::utilities::check_invocable;
using mimkl::utilities::print_type;

//#define SPDLOG_DEBUG_ON
//#define SPDLOG_TRACE_ON

// Compile time log levels
// define SPDLOG_DEBUG_ON or SPDLOG_TRACE_ON
// SPDLOG_TRACE(console, "Enabled only #ifdef SPDLOG_TRACE_ON..{} ,{}",
// 1, 3.23);  SPDLOG_DEBUG(console, "Enabled only #ifdef SPDLOG_DEBUG_ON.. {}
// ,{}", 1, 3.23);

int main(int argc, char **argv)
{

    // Runtime log levels
    spdlog::set_level(spdlog::level::trace); // Set global log level to info
    auto console = spdlog::stdout_color_mt("console");

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

    MATRIX(double) K(rows, rows);
    console->info("\e[1;32mtype: \e[0m \n{}",
                  print_type<decltype(k_poly(X, X, L))>());
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
    std::vector<MATRIX(double)> kernel_vec;
    kernel_vec.reserve(2);

    MATRIX(double) K2 = function_vec[0](X, X);
    console->info("a kernel from function (linear kernel with identity):\n{}",
                  K2);
    kernel_vec.push_back(K2);

    MATRIX(double) K_norm = mimkl::linear_algebra::normalize_kernel(K2);
    console->info("normalized above kernel:\n{}", K_norm);

    MATRIX(double) X_norm = X;
    X_norm.rowwise().normalize();
    console->info("normalized above kernel by means of centralizing the "
                  "original "
                  "data (works for linear kernel as feature space is original "
                  "space):\n{}",
                  function_vec[0](X_norm, X_norm));
    console->trace("where the normalized data is:\n{}", X_norm);

    assert(((K_norm - function_vec[0](X_norm, X_norm)).norm() <= 0.0000001) &&
           "normalize_kernels not equal to normalization in original space for "
           "linear kernel");

    assert(
    ((K_norm - mimkl::linear_algebra::normalize_kernel_prediction(K2, K2, K2))
     .norm() == 0.0) &&
    "normalize_kernels not equal to normalize_kernels_prediction");

    MATRIX(double) K_center = mimkl::linear_algebra::centralize_kernel(K2);
    console->info("centralized above kernel:\n{}", K_center);

    MATRIX(double) X_center = X;
    X_center = X.rowwise() - X.colwise().mean(); // in each dimension remove mean
    console->info("centralized above kernel by means of centralizing the "
                  "original data (works for linear kernel as feature space is "
                  "original space):\n{}",
                  function_vec[0](X_center, X_center));
    console->trace("where the centered data is:\n{}", X_center);

    assert(
    ((K_center - function_vec[0](X_center, X_center)).norm() == 0.0) &&
    "centralize_kernels not equal to centralization in original space for "
    "linear kernel");

    console->info("centralized normalized_kernel:\n{}",
                  mimkl::linear_algebra::centralize_kernel(K_norm));
    console->info("normalized centralized_kernel:\n{}",
                  mimkl::linear_algebra::normalize_kernel(K_center));

    MATRIX(double) K3 = function_vec[1](X, X);
    console->info("another kernel from function:\n{}", K3);
    kernel_vec.push_back(K3);

    MATRIX(double) kernel_sum_reference(rows, rows);
    kernel_sum_reference << 4., 8., 10., 10., 0., 4., 8., 16., 20., 20., 0., 8.,
    10., 20., 25., 25., 0., 10., 10., 20., 25., 25., 0., 10., 0., 0., 0., 0.,
    0., 0., 4., 8., 10., 10., 0., 4.;
    // TODO normalize samples/Kernel/kernel_trace/...

    spdlog::stdout_color_mt("lin_alg");

    COLUMN(double) c = COLUMN(double)::Constant(function_vec.size(), 1);
    MATRIX(double)
    kernel_weighted_sum =
    mimkl::linear_algebra::aggregate_weighted_kernels(X, X, function_vec, c);
    MATRIX(double)
    kernel_sum = mimkl::linear_algebra::aggregate_kernels(X, X, function_vec);
    console->trace("sum of kernels:\n{}", kernel_sum);
    console->trace("sum of (same) weighted kernels:\n{}", kernel_weighted_sum);

    assert(((kernel_sum_reference - kernel_sum).norm() == 0.0) &&
           "aggregate_kernels");
    assert(((kernel_sum_reference - kernel_weighted_sum).norm() == 0.0) &&
           "aggregate_weighted_kernels");

    //  test mapping

    double pi = 3.14159265358979323846; // std::atan(1.)*4. ;
    console->info(
    "testing/debugging the mapping between eigen and dlib, pi: {}", pi);
    COLUMN(double) to_map = COLUMN(double)::Constant(4, pi);
    console->info("initial (to_map):\n{}\n sum: {}", to_map, to_map.sum());
    dlib::matrix<double, 0, 1> mapped = dlib::mat(to_map);
    change_dlib_mat(mapped);
    console->info("mapped and assigned to dlib (mapped):\n{}\n sum: {}", mapped,
                  dlib::sum(mapped));
    console->info("change in original? (to_map):\n{}\n sum: {}", to_map,
                  to_map.sum());
    //  change_dlib_mat(dlib::mat(to_map));
    //  console->info("change in original when mapping changed without
    //  assignment?
    //  (to_map):\n{}\n sum: {}", to_map, to_map.sum());

    COLUMN(double) back_mapped = mimkl::linear_algebra::dlib_to_eigen(mapped);
    console->info("backmapped with assignment (back_mapped):\n{}\n sum: {}",
                  back_mapped, back_mapped.sum());

    MATRIX(double) M;
    mimkl::linear_algebra::squared_euclidean_distances(X, M);
    M = M.array().sqrt();
    console->info("euclidean distances:\n{}", M);

    return EXIT_SUCCESS;
}
