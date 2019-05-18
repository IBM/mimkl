#ifndef INCLUDE_MIMKL_KERNELS_KERNELS_HANDLER_HPP_
#define INCLUDE_MIMKL_KERNELS_KERNELS_HANDLER_HPP_

#include <functional>
#include <memory>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/utilities.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

using mimkl::definitions::Index;

namespace mimkl
{
namespace kernels_handler
{

static std::shared_ptr<spdlog::logger> logger =
spdlog::stdout_color_mt("KernelsHandler");

template <typename Scalar, typename Kernel>
class KernelsHandler
{
    private:
    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;
    std::shared_ptr<spdlog::logger> _logger = spdlog::get("KernelsHandler");

    bool _precompute;
    bool _trace_normalization;

    Index _number_of_kernels = 0;

    std::shared_ptr<Matrix> _lhs_ptr = nullptr;
    std::shared_ptr<Matrix> _rhs_ptr = nullptr;

    Index _lhs_size = 0;
    Index _rhs_size = 0;

    std::vector<Kernel> _kernel_functions;
    std::vector<Matrix> _kernel_matrices;
    std::vector<Scalar> _trace_factors;

    // defining aggregational behaviour
    std::function<Matrix(const Index)> _square_brakets;
    std::function<Matrix()> _sum;
    std::function<Matrix(const Column &)> _weighted_sum;

    void precompute_kernel_matrices(const bool, const bool);
    void set_lambdas_to_throw();
    void set_lambdas(const bool, const bool, const bool);

    public:
    // constructors
    KernelsHandler() = default;
    KernelsHandler(const std::vector<Kernel> &kernel_functions,
                   const bool precompute = true,
                   const bool trace_normalization = true)
    : _precompute(precompute), _trace_normalization(trace_normalization)
    {
        set_lambdas_to_throw();
        if (kernel_functions.size() > 0)
            set_functions(kernel_functions, trace_normalization);
    }

    KernelsHandler(const std::vector<Matrix> &kernel_matrices,
                   const bool precompute = true,
                   const bool trace_normalization = true)
    : _trace_normalization(trace_normalization)
    {
        set_lambdas_to_throw();
        _precompute = true;
        if (kernel_matrices.size() > 0)
            set_matrices(kernel_matrices, trace_normalization);
    }

    // getters
    bool get_precompute() const;
    bool get_trace_normalization() const;
    Index get_number_of_kernels() const;
    Matrix get_lhs() const;
    Index get_lhs_size() const;
    Index get_rhs_size() const;
    std::vector<Kernel> get_functions() const;
    std::vector<Matrix> get_matrices() const;
    std::vector<Scalar> get_trace_factors() const;

    // setters
    void set_lhs(const Matrix &);
    void set_rhs(const Matrix &);
    void
    set_matrices(const std::vector<Matrix> &, const bool learning_mode = false);
    void set_functions(const std::vector<Kernel> &,
                       const bool learning_mode = false);
    void set_trace_factors(const std::vector<Scalar> &);

    Matrix operator[](const Index i) { return _square_brakets(i); }
    Matrix sum() { return _sum(); }
    Matrix sum(const Column &kernels_weights)
    {
        return _weighted_sum(kernels_weights);
    }
    Column get_corrected_kernels_weights(Column kernels_weights)
    {
        for (Index i = 0; i < _trace_factors.size(); ++i)
        {
            kernels_weights(i) /= _trace_factors[i];
        }
        return kernels_weights;
    }
};

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_lambdas_to_throw()
{
    _logger->debug("set_lambdas_to_throw() start");
    _square_brakets = [](const Index i) {
        throw std::logic_error("Kernels are not available");
        return Matrix();
    };
    _sum = []() {
        throw std::logic_error("Kernels are not available");
        return Matrix();
    };
    _weighted_sum = [](const Column &c) {
        throw std::logic_error("Kernels are not available");
        return Matrix();
    };
    _logger->debug("set_lambdas_to_throw() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_lambdas(const bool precompute,
                                                 const bool trace_normalization,
                                                 const bool learning_mode)
{
    _logger->debug("set_lambdas() start");
    _logger->debug("lhs samples: {}",
                   (_lhs_ptr != nullptr) ? _lhs_ptr->rows() : 0);
    _logger->debug("rhs samples: {}",
                   (_rhs_ptr != nullptr) ? _rhs_ptr->rows() : 0);
    if (precompute)
    { // trace_normalization or not already applied
        _square_brakets = [this](const Index i) { return _kernel_matrices[i]; };
        _sum = [this]() {
            return mimkl::linear_algebra::sum_matrices(_kernel_matrices);
        };
        _weighted_sum = [this](const Column &kernels_weights) {
            return mimkl::linear_algebra::sum_weighted_matrices(_kernel_matrices,
                                                                kernels_weights);
        };
    }
    else
    { // "lazy", compute (and trace_normalization) anew on each access
        if (trace_normalization)
        {
            if (learning_mode)
            { // learn trace_factor
                _trace_factors.resize(_number_of_kernels);
                _square_brakets = [this](const Index i) {
                    Matrix K = _kernel_functions[i](*_lhs_ptr, *_rhs_ptr);
                    _trace_factors[i] = mimkl::linear_algebra::trace_normalize(K);
                    return K;
                };
            }
            else
            { // apply learned trace_normalization
                if (_trace_factors.size() != _number_of_kernels)
                    throw std::logic_error("The trace_factors have not been "
                                           "learned. Sizes don't match");
                _square_brakets = [this](const Index i)
                -> Matrix { // enforce return type because deduction
                    // can't be done in the body
                    return _kernel_functions[i](*_lhs_ptr, *_rhs_ptr) /
                           _trace_factors[i];
                };
            }
            _sum = [this, learning_mode]() {
                return mimkl::linear_algebra::aggregate_trace_normalized_kernels(
                *_lhs_ptr, *_rhs_ptr, _kernel_functions, _trace_factors,
                learning_mode);
            };
            _weighted_sum = [this, learning_mode](const Column &kernels_weights) {
                return mimkl::linear_algebra::aggregate_weighted_trace_normalized_kernels(
                *_lhs_ptr, *_rhs_ptr, _kernel_functions, _trace_factors,
                learning_mode, kernels_weights);
            };
        }
        else
        { // no trace_normalization
            _square_brakets = [this](const Index i) {
                return _kernel_functions[i](*_lhs_ptr, *_rhs_ptr);
            };
            _sum = [this]() {
                return mimkl::linear_algebra::aggregate_kernels(
                *_lhs_ptr, *_rhs_ptr, _kernel_functions);
            };
            _weighted_sum = [this](const Column &kernels_weights) {
                return mimkl::linear_algebra::aggregate_weighted_kernels(
                *_lhs_ptr, *_rhs_ptr, _kernel_functions, kernels_weights);
            };
        }
    }
    _logger->debug("set_lambdas() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::precompute_kernel_matrices(
const bool trace_normalization, const bool learning_mode)
{
    _logger->debug("precompute_kernel_matrices() start");
    if (trace_normalization && !learning_mode &&
        _trace_factors.size() != _number_of_kernels)
        throw std::logic_error("The trace_factors must first be learned");
    _kernel_matrices.resize(_number_of_kernels);

    if (trace_normalization)
    {
        if (learning_mode)
        { // learn and apply trace_factors
            _trace_factors.resize(_number_of_kernels);
            _logger->debug("trace_normalization true, is_symmetric true");
            for (Index i = 0; i < _number_of_kernels; ++i)
            {
                _kernel_matrices[i] = _kernel_functions[i](*_lhs_ptr, *_rhs_ptr);
                _trace_factors[i] =
                mimkl::linear_algebra::trace_normalize(_kernel_matrices[i]);
                _logger->trace("precomputed kernel {}, trace is {}, "
                               "trace_factor is {}",
                               i, _kernel_matrices[i].trace(), _trace_factors[i]);
            }
        }
        else
        { // apply learned trace_factors
            _logger->debug("trace_normalization true, is_symmetric false");
            for (Index i = 0; i < _number_of_kernels; ++i)
            {
                _kernel_matrices[i] = _kernel_functions[i](*_lhs_ptr, *_rhs_ptr);
                _kernel_matrices[i] /= _trace_factors[i];
                _logger->trace("precomputed kernel {}, trace is {}, corrected "
                               "by factor {}",
                               i, _kernel_matrices[i].trace(), _trace_factors[i]);
            }
        }
    }
    else
    {
        _logger->debug("trace_normalization false");
        for (Index i = 0; i < _number_of_kernels; ++i)
        {
            _kernel_matrices[i] = _kernel_functions[i](*_lhs_ptr, *_rhs_ptr);
            _logger->trace("precomputed kernel {}, trace is {}", i,
                           _kernel_matrices[i].trace());
        }
    }
    _logger->debug("precompute_kernel_matrices() done");
}

template <typename Scalar, typename Kernel>
typename KernelsHandler<Scalar, Kernel>::Matrix
KernelsHandler<Scalar, Kernel>::get_lhs() const
{
    return (_lhs_ptr != nullptr) ? Matrix(*_lhs_ptr.get()) : Matrix();
}

template <typename Scalar, typename Kernel>
Index KernelsHandler<Scalar, Kernel>::get_lhs_size() const
{
    return _lhs_size;
}

template <typename Scalar, typename Kernel>
Index KernelsHandler<Scalar, Kernel>::get_rhs_size() const
{
    return _rhs_size;
}

template <typename Scalar, typename Kernel>
bool KernelsHandler<Scalar, Kernel>::get_precompute() const
{
    return _precompute;
}

template <typename Scalar, typename Kernel>
bool KernelsHandler<Scalar, Kernel>::get_trace_normalization() const
{
    return _trace_normalization;
}

template <typename Scalar, typename Kernel>
Index KernelsHandler<Scalar, Kernel>::get_number_of_kernels() const
{
    return _number_of_kernels;
}

template <typename Scalar, typename Kernel>
std::vector<Kernel> KernelsHandler<Scalar, Kernel>::get_functions() const
{
    return _kernel_functions;
}

template <typename Scalar, typename Kernel>
std::vector<typename KernelsHandler<Scalar, Kernel>::Matrix>
KernelsHandler<Scalar, Kernel>::get_matrices() const
{
    return _kernel_matrices;
}

template <typename Scalar, typename Kernel>
std::vector<Scalar> KernelsHandler<Scalar, Kernel>::get_trace_factors() const
{
    return _trace_factors;
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_lhs(const Matrix &lhs)
{
    _logger->debug("set_lhs() start");
    if (_kernel_functions.size() == 0)
        throw std::logic_error("Kernel functions were not provided");
    if (_kernel_functions.size() != _number_of_kernels)
        throw std::logic_error(
        "Kernel functions are not matching the number of kernels");
    // make sure both sides are released
    _lhs_ptr.reset();
    _rhs_ptr.reset();
    _lhs_ptr = std::make_shared<Matrix>(lhs);
    _rhs_ptr = std::shared_ptr<Matrix>(_lhs_ptr);
    _lhs_size = lhs.rows();
    _rhs_size = lhs.rows();
    precompute_kernel_matrices(_trace_normalization, _trace_normalization);
    set_lambdas(_precompute, _trace_normalization, _trace_normalization);
    _logger->debug("set_lhs() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_rhs(const Matrix &rhs)
{
    _logger->debug("set_rhs() start");
    if (_kernel_functions.size() == 0)
        throw std::logic_error("Kernel functions were not provided");
    if (_kernel_functions.size() != _number_of_kernels)
        throw std::logic_error(
        "Kernel functions are not matching the number of kernels");
    // make sure the old object is released
    _rhs_ptr.reset();
    _rhs_ptr = std::make_shared<Matrix>(rhs);
    _rhs_size = rhs.rows();
    if (_lhs_ptr == nullptr)
    {
        set_lambdas_to_throw();
        return;
    }
    precompute_kernel_matrices(_trace_normalization, false);
    set_lambdas(_precompute, _trace_normalization, false);
    _logger->debug("set_rhs() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_matrices(
const std::vector<Matrix> &kernel_matrices, const bool learning_mode)
{
    _logger->debug("set_matrices() start");
    if (learning_mode && kernel_matrices[0].rows() != kernel_matrices[0].cols())
        throw std::logic_error("The matrices must be squared");
    // NOTE: think about the reset
    // reset functions
    // std::vector<Kernel>().swap(_kernel_functions);
    _kernel_matrices = kernel_matrices;
    _number_of_kernels = _kernel_matrices.size();
    _lhs_size = _kernel_matrices[0].rows();
    _rhs_size = _kernel_matrices[0].cols();
    if (_trace_normalization)
    {
        if (learning_mode)
        {
            // learn trace_factors and apply
            _trace_factors.resize(_number_of_kernels);
            for (Index i = 0; i < _number_of_kernels; ++i)
            {
                _trace_factors[i] =
                mimkl::linear_algebra::trace_normalize(_kernel_matrices[i]);
            }
        }
        else
        {
            if (_trace_factors.size() != _number_of_kernels)
                throw std::logic_error(
                "The trace_factors have not been learned. Sizes don't match");
            // apply trace_factors
            for (Index i = 0; i < _number_of_kernels; ++i)
            {
                _kernel_matrices[i] /= _trace_factors[i];
            }
        }
    }
    set_lambdas(true, _trace_normalization, learning_mode);
    _logger->debug("set_matrices() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_functions(
const std::vector<Kernel> &kernel_functions, const bool learning_mode)
{
    _logger->debug("set_functions() start");
    // NOTE: think about the reset
    // reset matrices
    // std::vector<Matrix>().swap(_kernel_matrices);
    _kernel_functions = kernel_functions;
    _number_of_kernels = _kernel_functions.size();
    _lhs_size = 0;
    _rhs_size = 0;
    set_lambdas(_precompute, _trace_normalization, learning_mode);
    _logger->debug("set_functions() done");
}

template <typename Scalar, typename Kernel>
void KernelsHandler<Scalar, Kernel>::set_trace_factors(
const std::vector<Scalar> &trace_factors)
{
    _trace_factors = trace_factors;
}

} // namespace kernels_handler
} // namespace mimkl

#endif /* INCLUDE_MIMKL_KERNELS_KERNELS_HANDLER_HPP_ \
        */
