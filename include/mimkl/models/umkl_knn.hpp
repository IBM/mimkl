#ifndef INCLUDE_MIMKL_MODELS_UMKL_KNN_HPP_
#define INCLUDE_MIMKL_MODELS_UMKL_KNN_HPP_

#include <dlib/optimization.h>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/model.hpp>
#include <queue>
#include <spdlog/spdlog.h>
#include <thread>
#include <type_traits>
#include <utility>

using dlib::mat;
using mimkl::definitions::Index;
using mimkl::kernels_handler::KernelsHandler;
using mimkl::utilities::sort_indices_decending;

namespace mimkl
{
namespace models
{

static std::shared_ptr<spdlog::logger> logger_umkl_knn =
spdlog::stdout_color_mt("UMKLKNN");

template <typename Scalar, typename Kernel>
void find_and_update_neighbours_using_kernels(
KernelsHandler<Scalar, Kernel> &kernels_handler,
const Index number_of_kernels,
const Index number_of_support_vectors,
const Index k,
MATRIX(Scalar) & W)
{
// for each kernel, get k-NN and add it to upper triangular part of W
// TODO: consider allocation of W_local outside of the loop
// currently not in place given SegFault in development
#pragma omp parallel for shared(kernels_handler, W)
    for (Index m = 0; m < number_of_kernels; ++m)
    {
        // set W_local copy
        MATRIX(Scalar)
        W_local = MATRIX(Scalar)::Zero(number_of_support_vectors,
                                       number_of_support_vectors);
        // TODO: think about how to avoid this copy
        MATRIX(Scalar) kernel = kernels_handler[m];
        for (Index c = 0; c < number_of_support_vectors; ++c)
        {
            // no self loops admitted in W we have to treat the handle the case
            // i==c see original implementation
            // https://github.com/cran/mixKernel/blob/master/R/combine.kernels.R
            // starting by filling a neighbors heap
            std::priority_queue<std::pair<Scalar, Index>,
                                std::vector<std::pair<Scalar, Index>>,
                                std::greater<std::pair<Scalar, Index>>>
            nn_heap;
            Index i = 0;
            // initialize initial k elements
            while (nn_heap.size() < k)
            {
                if (i != c)
                {
                    nn_heap.push(std::pair<Scalar, Index>(kernel(i, c), i));
                }
                // keep track of the index
                ++i;
            }
            // compute k nearest neighbors for c
            for (; i < number_of_support_vectors; ++i)
            {
                if (i != c && nn_heap.top().first < kernel(i, c))
                {
                    nn_heap.pop();
                    nn_heap.push(std::pair<Scalar, Index>(kernel(i, c), i));
                }
            }
            // consuming the neighbors heap and populating the knn graph
            while (!nn_heap.empty())
            {
                Index neighbor = nn_heap.top().second;
                neighbor > c ? W_local(c, neighbor) = 1 :
                               W_local(neighbor, c) = 1;
                nn_heap.pop();
            }
        }
// update W with W_local
#pragma omp critical
        {
            W.template triangularView<Eigen::Upper>() += W_local;
        }
    }
}

//! Unsupervised Multiple Kernel Learning (2017), notion of original topology
//! from k-NN of kernels
/*! (sparse version from paper)
 *
 */
template <typename Scalar, typename Kernel>
class UMKLKNN : public Model<Scalar, Kernel>
{

    private:
    // inheritance of templatized base members
    using Model<Scalar, Kernel>::_kernels_handler;
    using Model<Scalar, Kernel>::_trained;
    using Model<Scalar, Kernel>::_precompute;
    using Model<Scalar, Kernel>::_trace_normalization;
    using Model<Scalar, Kernel>::_number_of_support_vectors;
    using Model<Scalar, Kernel>::_number_of_kernels;

    std::shared_ptr<spdlog::logger> _logger = spdlog::get("UMKLKNN");

    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;
    typedef Eigen::Map<const Column> MapColumn;
    typedef dlib::matrix<Scalar, 0, 1, dlib::default_memory_manager, dlib::column_major_layout>
    DlibColumn;

    Index _k = 5; // for k nearest neighbors (k-NN)
    double _epsilon = 0.0001;
    Index _maxiter_qp = 100000;

    Matrix _W;
    Matrix _S;

    DlibColumn _beta_dlib;
    Column _beta;

    void compute_w_matrix();
    void compute_s_matrix();
    void optimize_beta();
    void fit();

    public:
    UMKLKNN(const std::vector<Kernel> & = std::vector<Kernel>(),
            const bool precompute = true,
            const bool trace_normalization = true,
            const Index k = 5,
            const double epsilon = 0.0001,
            const Index maxiter_qp = 100000);
    UMKLKNN(const std::vector<Matrix> & = std::vector<Matrix>(),
            const bool precompute = true,
            const bool trace_normalization = true,
            const Index k = 5,
            const double epsilon = 0.0001,
            const Index maxiter_qp = 100000);

    void fit(const Matrix &);
    void fit(const std::vector<Matrix> &);
    Matrix predict(const Matrix &);
    Matrix predict(const std::vector<Matrix> &);

    Index get_k() const;
    double get_epsilon() const;
    Index get_maxiter_qp() const;
    Column get_beta() const;
    Matrix get_optimal_kernel();

    void set_k(const Index);
    void set_epsilon(const double);
    void set_maxiter_qp(const Index);
    void set_parameters(const Index,
                        const double epsilon = 0.0001,
                        const Index maxiter_qp = 100000);
    void set_beta(const Column &);
};

template <typename Scalar, typename Kernel>
UMKLKNN<Scalar, Kernel>::UMKLKNN(const std::vector<Kernel> &kernel_functions,
                                 const bool precompute,
                                 const bool trace_normalization,
                                 const Index k,
                                 const double epsilon,
                                 const Index maxiter_qp)
: Model<Scalar, Kernel>(kernel_functions, precompute, trace_normalization)
{
    _beta = Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
    _beta_dlib = mat(_beta);
    set_parameters(k, epsilon, maxiter_qp);
}

template <typename Scalar, typename Kernel>
UMKLKNN<Scalar, Kernel>::UMKLKNN(const std::vector<Matrix> &kernel_matrices,
                                 const bool precompute,
                                 const bool trace_normalization,
                                 const Index k,
                                 const double epsilon,
                                 const Index maxiter_qp)
: Model<Scalar, Kernel>(kernel_matrices, precompute, trace_normalization)
{
    _beta = Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
    _beta_dlib = mat(_beta);
    set_parameters(k, epsilon, maxiter_qp);
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::fit()
{
    _number_of_support_vectors = _kernels_handler.get_lhs_size();
    _number_of_kernels = _kernels_handler.get_number_of_kernels();
    compute_w_matrix();
    compute_s_matrix();
    optimize_beta();
    _trained = true;
    _logger->debug("all done");
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::fit(const std::vector<Matrix> &kernel_matrices)
{
    _kernels_handler.set_matrices(kernel_matrices, true);
    fit();
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::fit(const Matrix &X)
{
    _kernels_handler.set_lhs(X);
    fit();
}

template <typename Scalar, typename Kernel>
typename UMKLKNN<Scalar, Kernel>::Matrix
UMKLKNN<Scalar, Kernel>::predict(const Matrix &X)
{
    if (!_trained)
        throw std::logic_error("The model should be trained first (after "
                               "instantiation or change in parameters)");
    _kernels_handler.set_rhs(X);
    return get_optimal_kernel();
}

template <typename Scalar, typename Kernel>
typename UMKLKNN<Scalar, Kernel>::Matrix
UMKLKNN<Scalar, Kernel>::predict(const std::vector<Matrix> &kernel_matrices)
{
    if (!_trained)
        throw std::logic_error("The model should be trained first (after "
                               "instantiation or change in parameters)");
    if (kernel_matrices[0].rows() != _number_of_support_vectors)
        throw std::length_error("Similarities must be provided for all support "
                                "vectors; matrices have wrong number of rows.");
    if (kernel_matrices.size() != _number_of_kernels)
        throw std::length_error(
        "Same number of kernels as on training is required");
    _kernels_handler.set_matrices(kernel_matrices);
    return get_optimal_kernel();
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::compute_w_matrix()
{
    // ensure a clean status
    _W.resize(_number_of_support_vectors, _number_of_support_vectors);
    _W.setZero();

    // find k-nearest neighbors filling upper part of _W
    find_and_update_neighbours_using_kernels(_kernels_handler, _number_of_kernels,
                                             _number_of_support_vectors, _k, _W);

    // symmetrize _W
    _W.template triangularView<Eigen::Lower>() = _W.transpose();
    // check if we handled properly the self loops
    if (_W.diagonal().sum() > 0.0)
    {
        _logger->critical("W diagonal contains non zero elements");
    }
    _logger->trace("Matrix W:\n{}", _W);
    _logger->debug("W done");
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::compute_s_matrix()
{
    _S.resize(_number_of_kernels, _number_of_kernels);
    _S.setZero();
    Matrix K_st(_number_of_support_vectors, _number_of_support_vectors);
    for (Index s = 0; s < _number_of_kernels; ++s)
    {
        for (Index t = s; t < _number_of_kernels; ++t)
        { // compute
            K_st = _kernels_handler[s] * _kernels_handler[t];
            _S(s, t) =
            ((_W * K_st.diagonal()).sum() - _W.cwiseProduct(K_st).sum());
            if (t > s)
            {
                _S(t, s) = _S(s, t);
            }
        }
    }
    _logger->trace("Matrix S:\n{}", _S);
    _logger->debug("S done");
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::optimize_beta()
{
    _beta = Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
    _beta_dlib = mat(_beta);
    Index iterations = dlib::solve_qp_using_smo(
    mat(_S), dlib::zeros_matrix<Scalar>(0, _number_of_kernels), _beta_dlib,
    _epsilon, _maxiter_qp); // 2*mat(_S) would be correct, but not needed
    if (iterations > _maxiter_qp)
    {
        _logger->critical("the qp-solver did not finish in {} iterations",
                          _maxiter_qp);
    }
    _logger->debug("{} qp iterations", iterations);
    _beta = mimkl::linear_algebra::dlib_to_eigen(_beta_dlib);
}

template <typename Scalar, typename Kernel>
Index UMKLKNN<Scalar, Kernel>::get_k() const
{
    return _k;
}

template <typename Scalar, typename Kernel>
double UMKLKNN<Scalar, Kernel>::get_epsilon() const
{
    return _epsilon;
}

template <typename Scalar, typename Kernel>
Index UMKLKNN<Scalar, Kernel>::get_maxiter_qp() const
{
    return _maxiter_qp;
}

template <typename Scalar, typename Kernel>
typename UMKLKNN<Scalar, Kernel>::Column UMKLKNN<Scalar, Kernel>::get_beta() const
{
    _logger->debug(
    "The model has not been fit. Maybe the parameters were changed?");
    return _beta;
}

template <typename Scalar, typename Kernel>
typename UMKLKNN<Scalar, Kernel>::Matrix
UMKLKNN<Scalar, Kernel>::get_optimal_kernel()
{
    return _kernels_handler.sum(_beta);
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::set_k(const Index k)
{
    _k = k;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::set_epsilon(const double epsilon)
{
    _epsilon = epsilon;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::set_maxiter_qp(const Index maxiter_qp)
{
    _maxiter_qp = maxiter_qp;
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::set_parameters(const Index k,
                                             const double epsilon,
                                             const Index maxiter_qp)
{
    _k = k;
    _epsilon = epsilon;
    _maxiter_qp = maxiter_qp;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
    _logger->debug("k: {}", _k);
    _logger->debug("epsilon: {}", _epsilon);
    _logger->debug("maxiter_qp: {}", _maxiter_qp);
}

template <typename Scalar, typename Kernel>
void UMKLKNN<Scalar, Kernel>::set_beta(const Column &beta)
{
    if (beta.rows() != _number_of_kernels)
        throw std::length_error(
        "passed beta does not have one weight for each kernel");
    _beta = beta;
    _beta_dlib = mat(_beta);
}

} // namespace models
} // namespace mimkl

#endif /* INCLUDE_MIMKL_MODELS_UMKL_KNN_HPP_ */
