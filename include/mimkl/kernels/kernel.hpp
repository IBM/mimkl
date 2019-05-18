#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <cstdarg>
#include <mimkl/definitions.hpp>

using mimkl::definitions::Index;

namespace mimkl
{
namespace kernel
{

//! The linear kernel with the extended kernel K = X*Y_t
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed (unconjugated) right hand side KxM data-matrix.
\returns kernel_matrix the similarity NxK matrix.
\sa test/linear_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived>
KDerived linear_kernel(LhsDerived &lhs, RhsDerived &rhs)
{
    return lhs * rhs.adjoint();
}

//! The polynomial kernel with the extended kernel K =
//! (X*Y_t + c)^p
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed (unconjugated)  right hand side KxM data-matrix.
\param degree polynomial degree.
\param offset "free parameter trading off the influence of higher-order versus
lower-order terms in the polynomial".
\returns kernel_matrix the similarity NxK matrix.
\sa TODO
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived>
KDerived polynomial_kernel(LhsDerived &lhs,
                           RhsDerived &rhs,
                           const double degree,
                           const double offset)
{
    return (linear_kernel<KDerived>(lhs, rhs).array() + offset).pow(degree).matrix();
}

//! The gaussian kernel  k = exp(-
//! (x-y)_t*(x-y) / ( 2*s^2 )).
/*!  This squared pairwise euclidean distance cannot be expressed in a concise
matrix multiplication.
The squared distance of xi to yj = \f$(x_i^T*x_i) -(x_i^T*y_j)
-(y_j^T*x_i) + (y_j^T*y_j) \f$
This means next to \f$XY^T\f$ (the linear kernel) only the diagonal entries of
\f$XX^T\f$ and \f$YY^T\f$ are needed.
\param lhs left hand side NxM data-matrix.
\param rhs untransposed  (unconjugated) right hand side KxM data-matrix.
\param sigma_square variance of the bell curve.
\returns kernel_matrix the similarity NxK matrix.
\sa gaussian_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived>
KDerived
gaussian_kernel(LhsDerived &lhs, RhsDerived &rhs, const double sigma_square)
{

    return (-(
            /*! -2* lhs_inducer_rhs only in case of scalar matrices.
             with complex numbers we need: -lhs_rhs
             -lhs_rhs.adjoint
             imaginary parts cancel out! */
            ((-2 * (lhs * rhs.adjoint()).real()).colwise() +
             lhs.rowwise().squaredNorm())
            .rowwise() +
            lhs.rowwise().squaredNorm().transpose()) /
            (2 * sigma_square))
    .array()
    .exp()
    .matrix();
}

//! The sigmoidal kernel with the extended kernel k = tanh(a*
//! (x_t*y) +b).
/*!
\param lhs left hand side NxM data-matrix.
\param rhs untransposed  (unconjugated) right hand side KxM data-matrix.
\param a
\param b
\returns kernel_matrix the similarity NxK matrix.
\sa test/sigmoidal_induction
*/
template <typename KDerived, typename LhsDerived, typename RhsDerived>
KDerived
sigmoidal_kernel(LhsDerived &lhs, RhsDerived &rhs, const double a, const double b)
{

    return (a * linear_kernel<KDerived>(lhs, rhs).array() + b).tanh().matrix();
}

} // namespace kernel
} // namespace mimkl

#endif /*_KERNEL_HPP_*/
