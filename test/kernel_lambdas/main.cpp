#include "spdlog/spdlog.h"
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
#include <utility>

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

    MATRIX(double) K(rows, rows);
    console->info("\e[1;32mtype: \e[0m \n{}",
                  print_type<decltype(k_poly(X, X, L))>());
    console->info("decltype argument evaluated? (no):\n{}", K);
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(k_poly));
    console->info("check_invocable? {}", check_invocable(k_poly));

    std::vector<InducedFunction> function_vec =
    mimkl::induction::inducer_combination(k_poly, inducer_vec);

    MATRIX(double) K2 = function_vec[0](X, X);
    console->info("a kernel from function:\n{}", K2);

    MATRIX(double) K3 = function_vec[1](X, X);
    console->info("another kernel from function:\n{}", K3);

    assert(((K_ref0 - K2).norm() == 0) && "Identity Inducer");
    assert(((K_ref1 - K3).norm() == 0) && "non-identity Inducer");

    console->info("\e[1;35m not callable"); //-----------
    console->info("int is not invocable! {}",
                  mimkl::utilities::is_invocable<int>::value);

    console->info("\e[1;35m function pointer"); //-----------
    // lambda to function pointer only without any captures
    void (*foo)(int) = [](int a) {};
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(foo));
    console->info("function pointer check_invocable?: {}", check_invocable(foo));
    console->info("function pointer is_function_t?: {}",
                  mimkl::utilities::is_function_t<decltype(foo)>::value);

    std::function<void(int)> bar = foo;
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(bar));
    console->info("function pointer cast to std::function invocable?: {}",
                  check_invocable(bar));

    console->info("\e[1;35m lambdas"); //-----------
    auto lam = []() {};
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(lam));
    console->info("is lambda invocable? {}",
                  mimkl::utilities::is_invocable<decltype(lam)>::value);
    console->info("\e[1;32mtype: \e[0m \n{}", print_type([]() {}));
    console->info("check_invocable? {}", check_invocable([]() {}));

    console->info("\e[1;35m function object"); //-----------
    struct FunctionObject
    {
        int operator()(int a) { return a; };
    };
    FunctionObject fo;
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(fo));
    console->info("is invocable? {}",
                  mimkl::utilities::is_invocable<FunctionObject>::value);
    console->info("check invocable {}", check_invocable(fo));

    console->info("\e[1;35m std::function<>"); //-----------
    //  console->info("\e[1;32mtype: \e[0m \n{}",
    //  print_type<mimkl::induction::induce_polynomial_kernel>());
    console->info("\e[1;32mtype: \e[0m \n{}",
                  print_type<std::function<void(int)>>());
    console->info("is invocable? {}",
                  mimkl::utilities::is_invocable<std::function<void(int)>>::value);

    console->info("\e[1;35m function_vec element"); //-----------
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(function_vec[0]));
    console->info("element check_invocable {}", check_invocable(function_vec[0]));
    console->info("element is invocable? {}",
                  mimkl::utilities::is_invocable<decltype(function_vec[0])>::value);
    console->info("element is_function? {}",
                  std::is_function<decltype(function_vec[0])>::value);
    console->info("is_function_t ? {}",
                  mimkl::utilities::is_function_t<decltype(function_vec[0])>::value);
    //  console->info("is invocable?
    //  {}",mimkl::utilities::is_invocable<decltype(mimkl::induction::inducer_combination)>::value);

    console->info("\e[1;35m function_vec::value_type"); //-----------
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(function_vec));
    console->info("\e[1;32mtype: \e[0m \n{}",
                  print_type<decltype(function_vec)::value_type>());
    console->info("is vector type std::function? {}",
                  std::is_function<decltype(function_vec)::value_type>::value);
    console->info(
    "is vector type invocable? {}",
    mimkl::utilities::is_invocable<decltype(function_vec)::value_type>::value);

    console->info("\e[1;35m Eigen::Matrix"); //-----------
    console->info("\e[1;32mtype: \e[0m \n{}", print_type(K2));
    console->info("\e[1;32mtype: \e[0m \n{}", print_type<MATRIX(double)>());
    console->info("is std::function? {}", std::is_function<decltype(K2)>::value);
    console->info("is invocable? {}",
                  mimkl::utilities::is_invocable<decltype(K2)>::value);
    console->info("check_invocable {}", check_invocable(K2));

    console->info(
    "\e[1;35m checkout move semantics of Eigen"); //-----------checkout move
    // semantics
    std::function<MATRIX(double)()> spaghetti_monster = [console]() {
        MATRIX(double) B = MATRIX(double)::Constant(3, 3, 42);
        auto B_ptr = (void *)B.data();
        console->info("pointer to B {}", B_ptr);
        return B;
    };
    std::function<MATRIX(double)()> illuminati = [console]() {
        return MATRIX(double)::Constant(3, 3, 1) +
               MATRIX(double)::Constant(3, 3, 22);
    };

    MATRIX(double) A = 10 * MATRIX(double)::Random(3, 3);
    console->info("A {}", A);
    console->info("type of A.data():  {}", print_type(A.data()));
    console->info("type of (void *)A.data():  {}", print_type((void *)A.data()));
    auto A_ptr = (void *)A.data();
    console->info("pointer to A {}", A_ptr);

    A = spaghetti_monster();
    console->info("nooo, the spaghetti_monster!");
    auto C_ptr = (void *)A.data();
    console->info("pointer to A, is B? {}", C_ptr);

    double *A1_ptr = A.data();
    console->info("pointer to A (same, eh) {}", (void *)A1_ptr);
    MATRIX(double) D;
    D = std::move(A);
    double *D_ptr = D.data();
    console->info("pointer to D, is moved A? {}", (void *)D_ptr);
    console->info("D \n{}", D);

    D = illuminati();
    console->info("nooo, the illuminati!");
    double *E_ptr = D.data();
    console->info("pointer to D, is prolly pointer to temporary of evalueted "
                  "expression?{}",
                  (void *)E_ptr);
    console->info("E \n{}", D);
}
