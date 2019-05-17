#ifndef INCLUDE_MIMKL_MODELS_AVERAGE_HPP_
#define INCLUDE_MIMKL_MODELS_AVERAGE_HPP_

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

static std::shared_ptr<spdlog::logger> logger_average =
spdlog::stdout_color_mt("AverageMKL");

//! average kernel weights
template <typename Scalar, typename Kernel>
class AverageMKL : public Model<Scalar, Kernel>
{

    private:
    // inheritance of templatized base members
    using Model<Scalar, Kernel>::_kernels_handler;
    using Model<Scalar, Kernel>::_trained;
    using Model<Scalar, Kernel>::_precompute;
    using Model<Scalar, Kernel>::_trace_normalization;
    using Model<Scalar, Kernel>::_number_of_support_vectors;
    using Model<Scalar, Kernel>::_number_of_kernels;

    std::shared_ptr<spdlog::logger> _logger = spdlog::get("AverageMKL");

    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;
    typedef Eigen::Map<const Column> MapColumn;
    typedef dlib::matrix<Scalar, 0, 1, dlib::default_memory_manager, dlib::column_major_layout>
    DlibColumn;

    Column _weights;

    void fit();

    public:
    AverageMKL(const std::vector<Kernel> & = std::vector<Kernel>(),
               const bool precompute = true,
               const bool trace_normalization = true);
    AverageMKL(const std::vector<Matrix> & = std::vector<Matrix>(),
               const bool precompute = true,
               const bool trace_normalization = true);

    void fit(const Matrix &);
    void fit(const std::vector<Matrix> &);
    Matrix predict(const Matrix &);
    Matrix predict(const std::vector<Matrix> &);

    Matrix get_optimal_kernel();

    Column get_weights() const;
    void set_weights(const Column &);
};

template <typename Scalar, typename Kernel>
AverageMKL<Scalar, Kernel>::AverageMKL(const std::vector<Kernel> &kernel_functions,
                                       const bool precompute,
                                       const bool trace_normalization)
: Model<Scalar, Kernel>(kernel_functions, precompute, trace_normalization)
{
    _weights =
    Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
}

template <typename Scalar, typename Kernel>
AverageMKL<Scalar, Kernel>::AverageMKL(const std::vector<Matrix> &kernel_matrices,
                                       const bool precompute,
                                       const bool trace_normalization)
: Model<Scalar, Kernel>(kernel_matrices, precompute, trace_normalization)
{
    _weights =
    Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
}

template <typename Scalar, typename Kernel>
void AverageMKL<Scalar, Kernel>::fit()
{
    _number_of_support_vectors = _kernels_handler.get_lhs_size();
    _number_of_kernels = _kernels_handler.get_number_of_kernels();

    if (_weights.size() != _number_of_kernels)
    {
        _weights =
        Column::Constant(_number_of_kernels, (double)1. / _number_of_kernels);
    }
    _trained = true;
    _logger->debug("all done");
}

template <typename Scalar, typename Kernel>
void AverageMKL<Scalar, Kernel>::fit(const std::vector<Matrix> &kernel_matrices)
{
    _kernels_handler.set_matrices(kernel_matrices, true);
    fit();
}

template <typename Scalar, typename Kernel>
void AverageMKL<Scalar, Kernel>::fit(const Matrix &X)
{
    _kernels_handler.set_lhs(X);
    fit();
}

template <typename Scalar, typename Kernel>
typename AverageMKL<Scalar, Kernel>::Matrix
AverageMKL<Scalar, Kernel>::predict(const Matrix &X)
{
    if (!_trained)
        throw std::logic_error("The model should be trained first (after "
                               "instantiation or change in parameters)");
    _kernels_handler.set_rhs(X);
    return get_optimal_kernel();
}

template <typename Scalar, typename Kernel>
typename AverageMKL<Scalar, Kernel>::Matrix
AverageMKL<Scalar, Kernel>::predict(const std::vector<Matrix> &kernel_matrices)
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
typename AverageMKL<Scalar, Kernel>::Column
AverageMKL<Scalar, Kernel>::get_weights() const
{
    if (!_trained)
        _logger->debug(
        "The model has not been fit. Maybe the parameters were changed?");
    return _weights;
}

template <typename Scalar, typename Kernel>
typename AverageMKL<Scalar, Kernel>::Matrix
AverageMKL<Scalar, Kernel>::get_optimal_kernel()
{
    return _kernels_handler.sum(_weights);
}

template <typename Scalar, typename Kernel>
void AverageMKL<Scalar, Kernel>::set_weights(const Column &weights)
{
    if (weights.rows() != _number_of_kernels)
        throw std::length_error(
        "passed weights does not have one weight for each kernel");
    _weights = weights;
}

} // namespace models
} // namespace mimkl

#endif /* INCLUDE_MIMKL_MODELS_AVERAGE_HPP_ */
