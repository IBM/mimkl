#ifndef INCLUDE_MIMKL_MODELS_MODEL_HPP_
#define INCLUDE_MIMKL_MODELS_MODEL_HPP_

#include <mimkl/definitions.hpp>
#include <mimkl/kernels_handler.hpp>

using mimkl::definitions::Index;
using mimkl::kernels_handler::KernelsHandler;

namespace mimkl
{
namespace models
{

template <typename Scalar, typename Kernel>
class Model
{ //<typename XDerived, typename LabelsDerived>
    private:
    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;

    protected:
    bool _trained = false;
    bool _precompute;
    bool _trace_normalization;
    KernelsHandler<Scalar, Kernel> _kernels_handler;
    Index _number_of_support_vectors = 0; // not initialized
    Index _number_of_kernels = 0;

    public:
    Model(const std::vector<Kernel> &kernel_functions = std::vector<Kernel>(),
          const bool precompute = true,
          const bool trace_normalization = true)
    : _precompute(precompute), _trace_normalization(trace_normalization)
    {
        _kernels_handler =
        KernelsHandler<Scalar, Kernel>(kernel_functions, precompute,
                                       trace_normalization);
        _number_of_kernels = _kernels_handler.get_number_of_kernels();
    }

    Model(const std::vector<Matrix> &kernel_matrices = std::vector<Matrix>(),
          const bool precompute = true,
          const bool trace_normalization = true)
    : _precompute(true), _trace_normalization(trace_normalization)
    {
        _kernels_handler = KernelsHandler<Scalar, Kernel>(kernel_matrices, true,
                                                          trace_normalization);
        _number_of_kernels = _kernels_handler.get_number_of_kernels();
    }

    Model(const KernelsHandler<Scalar, Kernel> &kernels_handler)
    {
        _kernels_handler = kernels_handler;
        _number_of_kernels = _kernels_handler.get_number_of_kernels();
        _precompute = _kernels_handler.get_precompute();
        _trace_normalization = _kernels_handler.get_trace_normalization();
    }

    // getters wrapping KernelsHandler ones
    std::vector<Kernel> get_functions() const;
    std::vector<Matrix> get_matrices() const;
    std::vector<Scalar> get_trace_factors() const;
    Matrix get_support_vectors() const;
    bool get_precompute() const;
    bool get_trace_normalization() const;
    // retrieve kernel_weights for use with untracenormalized kernels
    Column get_corrected_kernels_weights(Column kernels_weights);

    // setters wrapping KernelsHandler ones
    void
    set_matrices(const std::vector<Matrix> &, const bool learning_mode = false);
    void set_functions(const std::vector<Kernel> &,
                       const bool learning_mode = false);
    void set_trace_factors(const std::vector<Scalar> &);
    void set_support_vectors(const Matrix &);


    virtual ~Model() = default;
};

template <typename Scalar, typename Kernel>
std::vector<Kernel> Model<Scalar, Kernel>::get_functions() const
{
    return _kernels_handler.get_functions();
}

template <typename Scalar, typename Kernel>
std::vector<typename Model<Scalar, Kernel>::Matrix>
Model<Scalar, Kernel>::get_matrices() const
{
    return _kernels_handler.get_matrices();
}

template <typename Scalar, typename Kernel>
std::vector<Scalar> Model<Scalar, Kernel>::get_trace_factors() const
{
    return _kernels_handler.get_trace_factors();
}

template <typename Scalar, typename Kernel>
typename Model<Scalar, Kernel>::Column
Model<Scalar, Kernel>::get_corrected_kernels_weights(Column kernels_weights)
{
    return _kernels_handler.get_corrected_kernels_weights(kernels_weights);
}


template <typename Scalar, typename Kernel>
typename Model<Scalar, Kernel>::Matrix
Model<Scalar, Kernel>::get_support_vectors() const
{
    return _kernels_handler.get_lhs();
}

template <typename Scalar, typename Kernel>
bool Model<Scalar, Kernel>::get_precompute() const
{
    return _kernels_handler.get_precompute();
}

template <typename Scalar, typename Kernel>
bool Model<Scalar, Kernel>::get_trace_normalization() const
{
    return _kernels_handler.get_trace_normalization();
}

template <typename Scalar, typename Kernel>
void Model<Scalar, Kernel>::set_functions(
const std::vector<Kernel> &kernel_functions, const bool learning_mode)
{
    _kernels_handler.set_functions(kernel_functions, learning_mode);
}

template <typename Scalar, typename Kernel>
void Model<Scalar, Kernel>::set_matrices(const std::vector<Matrix> &kernel_matrices,
                                         const bool learning_mode)
{
    _kernels_handler.set_matrices(kernel_matrices, learning_mode);
}

template <typename Scalar, typename Kernel>
void Model<Scalar, Kernel>::set_trace_factors(const std::vector<Scalar> &trace_factors)
{
    _kernels_handler.set_trace_factors(trace_factors);
}

template <typename Scalar, typename Kernel>
void Model<Scalar, Kernel>::set_support_vectors(const Matrix &support_vectors)
{
    _kernels_handler.set_lhs(support_vectors);
    _number_of_support_vectors = _kernels_handler.get_lhs_size();
}


} // namespace models
} // namespace mimkl
#endif /* INCLUDE_MIMKL_MODELS_MODEL_HPP_ */
