#ifndef _KERNEL_INDUCTION_HPP_
#define _KERNEL_INDUCTION_HPP_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <cstdarg>
#include <mimkl/definitions.hpp>
#include <spdlog/spdlog.h>

using mimkl::definitions::Index;

namespace mimkl
{
namespace induction
{

auto logger = spdlog::stdout_color_mt("kernel_induction");

//! If only the diagonal entries of \f$XLX^T\f$ are needed
/*!  one full matrix multiplication cannot be avoided with matrix induction,
    but we only compute the diagonal of \f$(X*L)*X^T\f$
\param matrix NxM data-matrix.
\param inducer pathway specific (sparse) MxM matrix.
\param diagonal a  Nx1 Matrix.
\sa test/diagonal_from_square_induction
*/
template <typename MatrixDerived, typename InducerDerived, typename DiagonalDerived>
void get_diagonal_from_square_induction(
const Eigen::MatrixBase<MatrixDerived> &matrix,
const Eigen::SparseMatrixBase<InducerDerived> &inducer,
const Eigen::EigenBase<DiagonalDerived> &diagonal)
{
    const Eigen::Matrix<typename MatrixDerived::Scalar, MatrixDerived::RowsAtCompileTime,
                        MatrixDerived::ColsAtCompileTime>
    matrix_inducer =
    matrix * inducer.template selfadjointView<Eigen::Lower>(); // no .noalias()
    // possible here:
    // matrix_inducer =
    // ;

    Eigen::EigenBase<DiagonalDerived> &diagonal_ =
    const_cast<Eigen::EigenBase<DiagonalDerived> &>(diagonal);
    const Index n = matrix_inducer.rows(); // == matrix.rows()
    if (diagonal.rows() < n)
    {
        spdlog::get("kernel_induction")
        ->critical("error in get_diagonal_from_square_induction():\n rows of "
                   "diagonal {}, rows needed: {}",
                   diagonal.rows(), matrix.rows());
    }
    //  try {
    for (Index i = 0; i < n; i++)
    {
        diagonal_.derived()(i, 0) =
        matrix_inducer.row(i) *
        matrix.adjoint().col(i); // matrix.row(i) for RealScalars
    }
    //  } catch (...){ //
    //		spdlog::get("kernel_induction")->critical("error in
    // get_diagonal_from_square_induction():\n rows of diagonal {}, rows needed:
    //[]", diagonal.rows(), matrix.rows());
    //		throw;
    //  }
}

//! Matrix induction of a linear kernel with the extended kernel K = X*L*Y_t
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed (unconjugated) right hand side KxM data-matrix.
\param inducer pathway specific (sparse, symmetric) MxM matrix.
\returns kernel_matrix the similarity NxK matrix.
\sa test/linear_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived, typename LDerived>
KDerived induce_linear_kernel(LhsDerived &lhs, RhsDerived &rhs, LDerived inducer)
{
    return lhs * inducer.template selfadjointView<Eigen::Lower>() *
           rhs.adjoint(); // symmetry allows optimization Lower vs. Upper?
}

//! Matrix induction of a polynomial kernel with the extended kernel K =
//! (X*L*Y_t + c)^p
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed (unconjugated)  right hand side KxM data-matrix.
\param inducer pathway specific (sparse) MxM matrix.
\param degree polynomial degree.
\param offset "free parameter trading off the influence of higher-order versus
lower-order terms in the polynomial".
\returns kernel_matrix the similarity NxK matrix.
\sa TODO
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived, typename LDerived>
KDerived induce_polynomial_kernel(LhsDerived &lhs,
                                  RhsDerived &rhs,
                                  LDerived inducer,
                                  const double degree,
                                  const double offset)
{
    return (induce_linear_kernel<KDerived>(lhs, rhs, inducer).array() + offset)
    .pow(degree)
    .matrix();
}

//! Matrix induction of a gaussian kernel with the extended kernel k = exp(-
//! (x-y)_t*L*(x-y) / ( 2*s^2 )).
/*!  This squared pairwise euclidean distance cannot be expressed in a concise
matrix multiplication.
The squared distance of xi to yj = \f$(x_i^T*L*x_i) -(x_i^T*L*y_j)
-(y_j^T*L*x_i) + (y_j^T*L*y_j) \f$
This means next to \f$XLY^T\f$ (the linear kernel) only the diagonal entries of
\f$XLX^T\f$ and \f$YLY^T\f$ are needed.
\param lhs left hand side NxM data-matrix.
\param rhs untransposed  (unconjugated) right hand side KxM data-matrix.
\param inducer pathway specific (sparse) MxM matrix.
\param sigma_square variance of the bell curve.
\returns kernel_matrix the similarity NxK matrix.
\sa gaussian_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived, typename LDerived>
KDerived induce_gaussian_kernel(LhsDerived &lhs,
                                RhsDerived &rhs,
                                LDerived inducer,
                                const double sigma_square)
{
    // we only need the diagonal entries of X*L*X_t, so only fill the diagonal
    // of
    // (X*L)*X_t
    Eigen::Matrix<typename LhsDerived::Scalar, LhsDerived::RowsAtCompileTime, 1>
    diag_lhs_inducer_lhs_t; // Length N
    if (LhsDerived::RowsAtCompileTime == Eigen::Dynamic)
        diag_lhs_inducer_lhs_t.resize(lhs.rows());
    get_diagonal_from_square_induction(lhs, inducer, diag_lhs_inducer_lhs_t);
    // same for (Y*L)*Y_t
    Eigen::Matrix<typename RhsDerived::Scalar, RhsDerived::RowsAtCompileTime, 1>
    diag_rhs_inducer_rhs_t; // Length K
    if (RhsDerived::RowsAtCompileTime == Eigen::Dynamic)
        diag_rhs_inducer_rhs_t.resize(rhs.rows());
    get_diagonal_from_square_induction(rhs, inducer, diag_rhs_inducer_rhs_t);

    // X*L*Y_t
    return (-(
            /*! -2* lhs_inducer_rhs only in case of scalar matrices.
             with complex numbers we need: -lhs_inducer_rhs
             -lhs_inducer_rhs.adjoint
             imaginary parts cancel out! */
            ((-2 * induce_linear_kernel<KDerived>(lhs, rhs, inducer).real()).colwise() +
             diag_lhs_inducer_lhs_t)
            .rowwise() +
            diag_rhs_inducer_rhs_t.transpose()) /
            (2 * sigma_square))
    .array()
    .exp()
    .matrix();
}

//! Matrix induction of a sigmoidal kernel with the extended kernel k = tanh(a*
//! (x_t*L*y) +b).
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed  (unconjugated) right hand side KxM data-matrix.
\param inducer pathway specific (sparse) MxM matrix.
\param a
\param b
\returns kernel_matrix the similarity NxK matrix.
\sa test/sigmoidal_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived, typename LDerived>
KDerived induce_sigmoidal_kernel(LhsDerived &lhs,
                                 RhsDerived &rhs,
                                 LDerived inducer,
                                 const double a,
                                 const double b)
{
    return (a * induce_linear_kernel<KDerived>(lhs, rhs, inducer).array() + b)
    .tanh()
    .matrix();
}

template <typename KDerived, typename LhsDerived, typename RhsDerived, typename LDerived>
std::vector<std::function<KDerived(LhsDerived &, RhsDerived &)>>
inducer_combination(std::function<KDerived(LhsDerived &, RhsDerived &, LDerived)>
                    kernel_function, // parameter specification done
                                     // previously
                    std::vector<LDerived> inducers)
{
    std::vector<std::function<KDerived(LhsDerived &, RhsDerived &)>> lambda_expressions;
    lambda_expressions.reserve(inducers.size());

    for (LDerived inducer : inducers)
    {
        lambda_expressions.push_back(
        [&, inducer](const LhsDerived &lhs, const RhsDerived &rhs) {
            return kernel_function(lhs, rhs, inducer);
        });
    }

    return lambda_expressions;
}

} // namespace induction
} // namespace mimkl

#endif /*_KERNEL_INDUCTION_HPP_*/
