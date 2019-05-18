#ifndef _LINEAR_ALGEBRA_HPP_
#define _LINEAR_ALGEBRA_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/StdVector>
#include <dlib/optimization.h>
#include <mimkl/definitions.hpp>
#include <mimkl/utilities.hpp>
#include <numeric>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using mimkl::definitions::Index;

namespace mimkl
{
namespace linear_algebra
{

//! \f$ K' = K / (tr(K) / n)\f$
/*!
devide matrix by a factor so that trace will equal the number of rows \f$ tr(K')
= n \f$
@param K square matrix to be trace_normalized by reference
@return trace_factor \f$ tr(K) / n\f$
*/
template <typename Derived>
typename Derived::Scalar trace_normalize(Derived &K)
{
    typename Derived::Scalar trace_factor = K.trace() / K.rows();
    K = K / trace_factor;
    return trace_factor;
}

//! \f$ k^{*}(x,y) = \fraq{k(x,y)}{\sqrt{k(x,x) k(y,y)}}\f$
template <typename Derived>
Derived normalize_kernel(const Derived &K)
{
    COLUMN(typename Derived::Scalar) diag = 1 / K.diagonal().array().sqrt();
    return K.array() * (diag * diag.transpose()).array(); // outer product of
                                                          // diagonal elements
}

//! \f$ k^{*}(x,y) = \fraq{k(x,y)}{\sqrt{k(x,x) k(y,y)}}\f$, where k(x,x) stems
//! from the training data and k(y,y) from the test data
template <typename LhsDerived, typename RhsDerived, typename Derived>
Derived normalize_kernel_prediction(const Derived &K,
                                    const LhsDerived &lhs,
                                    const RhsDerived &rhs)
{
    COLUMN(typename Derived::Scalar)
    diag_lhs = 1 / lhs.diagonal().array().sqrt();
    COLUMN(typename Derived::Scalar)
    diag_rhs = 1 / rhs.diagonal().array().sqrt();
    return K.array() *
           (diag_lhs * diag_rhs.transpose()).array(); // outer product of
                                                      // diagonal elements
}

//! centralizing the data in feature space \f$ \mathbf{K}' = ( \mathbf{I} -
//! \mathbf{1_N}) \mathbf{K} ( \mathbf{I} - \mathbf{1_N}) =  \mathbf{K} -
//! \mathbf{1_N} \mathbf{K} - \mathbf{K} \mathbf{1_N} + \mathbf{1_N} \mathbf{K}
//! \mathbf{1_N}\f$
/*!
where \f$\mathbf{1_N}\f$ denotes a N-by-N matrix for which each element takes
value 1/N
*/
template <typename Derived>
Derived centralize_kernel(const Derived &K)
{ // TODO how to do for test data?
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 1> rowwise_mean =
    K.rowwise().mean();
    return ((K.colwise() - rowwise_mean).rowwise() - rowwise_mean.transpose())
           .array() +
           rowwise_mean.mean();
}

template <typename Scalar,
          typename StorageOrder> // TODO why does const & not compile?
Eigen::Map<const COLUMN(Scalar)> dlib_to_eigen(
const dlib::matrix<Scalar, 0, 1, dlib::default_memory_manager, StorageOrder> &vector)
{ // more general case
    // with template
    // specializations not
    // needed as of yet.
    spdlog::get("console")->debug("sum over passed vector:\n{}",
                                  dlib::sum(vector));
    typedef COLUMN(Scalar) Column;
    typedef Eigen::Map<const Column> MapColumn;
    // Map Dlib to Eigen
    const Scalar *dlib_ptr = vector.begin(); //&vector.data(0,0);
    if (dlib::is_row_major(vector) ^ (Column::Flags & Eigen::RowMajorBit))
    { // if XOR warn, for instance Y: Y.isRowMajor
        auto logger = spdlog::get("console");
        logger->critical("make sure LibMat is in col-major or that mapping "
                         "happens "
                         "in row-major:\n"
                         "dlib::is_row_major(vector): {}\n"
                         "Column::Options: {}\n"
                         "(Column::Flags&Eigen::RowMajorBit): {}",
                         dlib::is_row_major(vector), Column::Options,
                         (Column::Flags & Eigen::RowMajorBit));
    };
    return MapColumn(dlib_ptr, vector.nr(), 1);
}

//! squared euclidean distance of rows in X to itself: M_ij =  x_i^T*x_i +
//! x_j^T*x_j -
//! 2*x_i^T*x_j
template <typename LhsDerived, typename MDerived>
void squared_euclidean_distances(const Eigen::MatrixBase<LhsDerived> &lhs,
                                 const Eigen::MatrixBase<MDerived> &distance_matrix)
{
    MDerived lin_kernel = lhs * lhs.transpose();
    const_cast<Eigen::MatrixBase<MDerived> &>(distance_matrix) =
    ((-2 * lin_kernel.real()).colwise() + lin_kernel.diagonal()).rowwise() +
    lin_kernel.diagonal().adjoint();
}

template <typename Scalar>
MATRIX(Scalar)
rowwise_soft_max(MATRIX(Scalar) M)
{
    for (Index i = 0; i < M.rows(); i++)
    {
        Scalar max = M.row(i).maxCoeff(); // numerical stability
        Scalar sum = 0;
        for (Index j = 0; j < M.cols(); j++)
        {
            sum += std::exp(M(i, j) - max);
        }
        Scalar normalizer = std::log(sum); // numerical stability
        for (Index j = 0; j < M.cols(); j++)
        {
            M(i, j) = std::exp(M(i, j) - max - normalizer);
        }
    }
    return M;
}

// TODO documentation
template <typename Scalar>
MATRIX(Scalar)
sum_matrices(const std::vector<MATRIX(Scalar)> &Ks)
{
    // fold over Ks
    return std::accumulate(Ks.begin(), Ks.end(),
                           MATRIX(Scalar)::Zero((*Ks.begin()).rows(),
                                                (*Ks.begin()).cols())
                           .eval()); //, [](a,b){operator+(a,b)}
}

template <typename Scalar>
MATRIX(Scalar)
sum_trace_normalized_matrices(const std::vector<MATRIX(Scalar)> &Ks,
                              std::vector<Scalar> &trace_factors)
{
    // fold over Ks, trace_normalize K before summing and store its trace
    MATRIX(Scalar) acc; // not writing 0s
    MATRIX(Scalar) K;   // temporary holding the trace_normalized kernel
    if (Ks.begin() != Ks.end())
    {
        Index i = 0;
        acc = Ks[i]; // non const copy
        trace_factors[i] = trace_normalize(acc);
        ++i;
        for (; i < Ks.size(); ++i)
        {
            K = Ks[i];
            trace_factors[i] = trace_normalize(K);
            acc += K;
        }
    }
}

// not using std::accumulate, see commented version below
template <typename Scalar, typename Function>
MATRIX(Scalar)
aggregate_kernels(const MATRIX(Scalar) & lhs,
                  const MATRIX(Scalar) & rhs,
                  const std::vector<Function> &Ks)
{
    MATRIX(Scalar) acc; // not writing 0s
    if (Ks.begin() != Ks.end())
    {
        typename std::vector<Function>::const_iterator iter = Ks.begin();
        acc = (*iter)(lhs, rhs);
        ++iter;
        for (; iter != Ks.end(); ++iter)
        {
            acc += (*iter)(lhs, rhs);
        }
    } // TODO else { throw | warn | ....}
    return acc;
}

template <typename Scalar, typename Function>
MATRIX(Scalar)
aggregate_trace_normalized_kernels(const MATRIX(Scalar) & lhs,
                                   const MATRIX(Scalar) & rhs,
                                   const std::vector<Function> &Ks,
                                   std::vector<Scalar> &trace_factors,
                                   bool update_trace_factors)
{
    if (Ks.size() != trace_factors.size())
        throw std::length_error(
        " number of functions does not match number of trace_factors");
    MATRIX(Scalar) acc; // not writing 0s
    if (update_trace_factors)
    {
        MATRIX(Scalar) K; // temporary holding the trace_normalized kernel
        if (Ks.begin() != Ks.end())
        {
            Index i = 0;
            K = Ks[i](lhs, rhs); // non const copy
            trace_factors[i] = trace_normalize(K);
            acc = K;
            ++i;
            for (; i < Ks.size(); ++i)
            {
                K = Ks[i](lhs, rhs);
                trace_factors[i] = trace_normalize(K);
                acc += K;
            }
        }
    }
    else
    {
        if (Ks.begin() != Ks.end())
        {
            Index i = 0;
            acc = Ks[i](lhs, rhs) / trace_factors[i];
            ++i;
            for (; i < Ks.size(); ++i)
            {
                acc += Ks[i](lhs, rhs) / trace_factors[i];
            }
        }
    }
    return acc;
}

// TODO documentation
template <typename Scalar>
MATRIX(Scalar)
sum_weighted_matrices(const std::vector<MATRIX(Scalar)> &Ks,
                      const COLUMN(Scalar) & eta)
{
    return std::inner_product(
    Ks.begin(), Ks.end(),
    eta.data(), // Eigen has no .begin() and .end()
    MATRIX(Scalar)::Zero((*Ks.begin()).rows(), (*Ks.begin()).cols()).eval());
}

template <typename Scalar, typename Function>
MATRIX(Scalar)
aggregate_weighted_kernels(const MATRIX(Scalar) & lhs,
                           const MATRIX(Scalar) & rhs,
                           const std::vector<Function> &fs,
                           const COLUMN(Scalar) & eta)
{
    MATRIX(Scalar) acc; // not writing 0s
    if (fs.begin() != fs.end())
    {
        Index i = 0;
        acc = fs[i](lhs, rhs) * eta(i);
        ++i;
        for (; i < fs.size(); ++i)
        {
            acc += fs[i](lhs, rhs) * eta(i);
        }
    } // TODO else { throw | warn | ....}
    return acc;
}

template <typename Scalar, typename Function>
MATRIX(Scalar)
aggregate_weighted_trace_normalized_kernels(const MATRIX(Scalar) & lhs,
                                            const MATRIX(Scalar) & rhs,
                                            const std::vector<Function> &Ks,
                                            std::vector<Scalar> &trace_factors,
                                            bool update_trace_factors, // update_trace_factors
                                                                       // or
                                            // read trace_factors
                                            const COLUMN(Scalar) & eta)
{
    if (Ks.size() != trace_factors.size())
        throw std::length_error(
        " number of functions does not match number of trace_factors");
    MATRIX(Scalar) acc; // not writing 0s
    if (update_trace_factors)
    {
        MATRIX(Scalar) K; // temporary holding the trace_normalized kernel
        if (Ks.begin() != Ks.end())
        {
            Index i = 0;
            K = Ks[i](lhs, rhs); // non const copy
            trace_factors[i] = trace_normalize(K);
            acc = K * eta(i);
            ++i;
            for (; i < Ks.size(); ++i)
            {
                K = Ks[i](lhs, rhs);
                trace_factors[i] = trace_normalize(K);
                acc += K * eta(i);
            }
        }
    }
    else
    {
        if (Ks.begin() != Ks.end())
        {
            Index i = 0;
            acc = Ks[i](lhs, rhs) / trace_factors[i] * eta(i);
            ++i;
            for (; i < Ks.size(); ++i)
            {
                acc += Ks[i](lhs, rhs) / trace_factors[i] * eta(i);
            }
        }
    } // TODO else { throw | warn | ....}
    return acc;
}

//! fill a sparse diagonal matrix with a given value
/*! only if elements don't exist already, else use coeffRef() instead of
insert()
\param I a sparse matrix of type Eigen::SparseMatrixBase<Derived>.
\param value a value to fill the diagonal of type double.
\sa test/sparse_diagonal
*/
template <typename Derived>
void fill_sparse_diagonal(Eigen::SparseMatrixBase<Derived> const &I,
                          const double value)
{
    // size of diagonal
    const Index &m = I.rows();
    const Index &n = I.cols();
    Index d = (m < n ? m : n);
    Eigen::SparseMatrixBase<Derived> &I_ =
    const_cast<Eigen::SparseMatrixBase<Derived> &>(I);
    for (Index i = 0; i < d; ++i)
    {
        I_.derived().insert(i, i) = value;
    }
}

// helpers for indexing an Eigen::Matrix
// adapted from
// http://eigen.tuxfamily.org/dox-devel/TopicCustomizing_NullaryExpr.html#title1
template <class ArgumentType, class RowIndexType, class ColumnIndexType>
class IndexingFunctor
{
    private:
    const ArgumentType &argument_;
    const RowIndexType &row_indices_;
    const ColumnIndexType &column_indices_;

    public:
    typedef Eigen::Matrix<
    typename ArgumentType::Scalar,
    RowIndexType::SizeAtCompileTime,
    ColumnIndexType::SizeAtCompileTime,
    ArgumentType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
    RowIndexType::MaxSizeAtCompileTime,
    ColumnIndexType::MaxSizeAtCompileTime>
    MatrixType;
    IndexingFunctor(const ArgumentType &argument,
                    const RowIndexType &row_indices,
                    const ColumnIndexType &column_indices)
    : argument_(argument),
      row_indices_(row_indices),
      column_indices_(column_indices)
    {
    }
    const typename ArgumentType::Scalar &
    operator()(Eigen::Index row, Eigen::Index column) const
    {
        return argument_(row_indices_[row], column_indices_[column]);
    }
};

template <class ArgumentType, class RowIndexType, class ColumnIndexType>
Eigen::CwiseNullaryOp<
IndexingFunctor<ArgumentType, RowIndexType, ColumnIndexType>,
typename IndexingFunctor<ArgumentType, RowIndexType, ColumnIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgumentType> &argument,
         const RowIndexType &row_indices,
         const ColumnIndexType &column_indices)
{
    typedef IndexingFunctor<ArgumentType, RowIndexType, ColumnIndexType> Indexer;
    typedef typename Indexer::MatrixType MatrixType;
    return MatrixType::NullaryExpr(row_indices.size(), column_indices.size(),
                                   Indexer(argument.derived(), row_indices,
                                           column_indices));
};
} // namespace linear_algebra
} // namespace mimkl

#endif /*_LINEAR_ALGEBRA_HPP_*/
