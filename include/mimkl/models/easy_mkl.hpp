#ifndef INCLUDE_MIMKL_MODELS_EASY_MKL_HPP_
#define INCLUDE_MIMKL_MODELS_EASY_MKL_HPP_

#include <Eigen/StdVector>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models/model.hpp>
#include <mimkl/solvers.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using mimkl::definitions::Indexing;

namespace mimkl
{
namespace models
{

static std::shared_ptr<spdlog::logger> logger_easy_mkl =
spdlog::stdout_color_mt("EasyMKL");

template <typename Scalar, typename Kernel>
class EasyMKL : public Model<Scalar, Kernel>
{

    private:
    // inheritance of templatized base members
    using Model<Scalar, Kernel>::_kernels_handler;
    using Model<Scalar, Kernel>::_trained;
    using Model<Scalar, Kernel>::_precompute;
    using Model<Scalar, Kernel>::_trace_normalization;
    using Model<Scalar, Kernel>::_number_of_support_vectors;
    using Model<Scalar, Kernel>::_number_of_kernels;

    // logger instance
    std::shared_ptr<spdlog::logger> _logger = spdlog::get("EasyMKL");
    // typedefs
    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;
    typedef ROW(Scalar) Row;
    typedef Eigen::Map<Column> MapColumn;

    // parameters
    double _lambda = 0.8;
    double _epsilon = 0.0001;
    bool _regularization_factor = false;

    Matrix _gammas;
    Matrix _etas;
    Row _biases;
    Indexing _class_map;

    std::vector<std::string> _unique_labels;
    Index _number_of_classes;
    Index _number_of_dichotomies;
    bool _binary;
    Matrix _kernels_sum;

    Column get_dichotomy(const std::string &);
    std::pair<Column, Column>
    optimize_gamma(const bool &, const Column &, const Column & = Column());
    Column compute_weights(const Column &);
    Scalar compute_bias(const Column &, const Column &, const Column &);

    void setup(std::vector<std::string>);
    void train(const Index &);
    void fit(const std::vector<std::string> &);
    Matrix decision_function();
    Column
    distances(const Column &, const Column &, const Column &, const Scalar &);

    public:
    EasyMKL(const std::vector<Kernel> & = std::vector<Kernel>(),
            const bool precompute = true,
            const bool trace_normalization = true,
            const double lambda = 0.8,
            const double epsilon = 0.0001,
            const bool regularization_factor = false);
    EasyMKL(const std::vector<Matrix> & = std::vector<Matrix>(),
            const bool precompute = true,
            const bool trace_normalization = true,
            const double lambda = 0.8,
            const double epsilon = 0.0001,
            const bool regularization_factor = false);

    // Functional, call with data matrix
    void fit(const Matrix &, const std::vector<std::string> &);
    std::vector<std::string> predict(const Matrix &);
    Matrix predict_proba(const Matrix &);
    Matrix decision_function(const Matrix &);
    // Matricial, call with kernel matrices
    void fit(const std::vector<Matrix> &, const std::vector<std::string> &);
    std::vector<std::string> predict(const std::vector<Matrix> &);
    Matrix predict_proba(const std::vector<Matrix> &);
    Matrix decision_function(const std::vector<Matrix> &);

    double get_lambda() const;
    double get_epsilon() const;
    bool get_regularization_factor() const;
    Matrix get_gammas() const;
    Matrix get_etas() const;
    Row get_biases() const;
    Indexing get_class_map() const;
    std::vector<std::string> get_one_versus_rest_order() const;
    Matrix get_optimal_kernel();
    Matrix get_optimal_kernel_by_class_index(const Index i);
    std::vector<Matrix> get_optimal_kernels();

    void set_lambda(const double);
    void set_epsilon(const double);
    void set_regularization_factor(const bool);
    void set_parameters(const double,
                        const double epsilon = 0.0001,
                        const bool regularization_factor = false);
    void set_gammas(const Matrix &);
    void set_etas(const Matrix &);
    void set_biases(const Row &);
    void set_class_map(const Indexing &);
};

template <typename Scalar, typename Kernel>
EasyMKL<Scalar, Kernel>::EasyMKL(const std::vector<Kernel> &kernel_functions,
                                 const bool precompute,
                                 const bool trace_normalization,
                                 const double lambda,
                                 const double epsilon,
                                 const bool regularization_factor)
: Model<Scalar, Kernel>(kernel_functions, precompute, trace_normalization)
{
    set_parameters(lambda, epsilon, regularization_factor);
}

template <typename Scalar, typename Kernel>
EasyMKL<Scalar, Kernel>::EasyMKL(const std::vector<Matrix> &kernel_matrices,
                                 const bool precompute,
                                 const bool trace_normalization,
                                 const double lambda,
                                 const double epsilon,
                                 const bool regularization_factor)
: Model<Scalar, Kernel>(kernel_matrices, precompute, trace_normalization)
{
    set_parameters(lambda, epsilon, regularization_factor);
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Column
EasyMKL<Scalar, Kernel>::get_dichotomy(const std::string &label)
{
    auto range = _class_map.equal_range(label);
    Column y = Column::Constant(_number_of_support_vectors, 1, -1);
    for (auto i = range.first; i != range.second; ++i)
    {
        y(i->second) = 1;
    }
    return y;
}

template <typename Scalar, typename Kernel>
std::pair<typename EasyMKL<Scalar, Kernel>::Column,
          typename EasyMKL<Scalar, Kernel>::Column>
EasyMKL<Scalar, Kernel>::optimize_gamma(const bool &regularization_factor,
                                        const Column &y,
                                        const Column &eta)
{
    mimkl::solvers::KOMD<Scalar> optimizer((eta.rows() > 0) ?
                                           _kernels_handler.sum(eta) :
                                           _kernels_sum,
                                           _lambda, _epsilon,
                                           regularization_factor);
    optimizer.solve(y);
    Column gamma = optimizer.get_result();
    return std::make_pair(gamma, y.cwiseProduct(gamma));
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Column
EasyMKL<Scalar, Kernel>::compute_weights(const Column &directed_gamma)
{
    // TODO maybe after all gammas are known and then compute etas rowwise, so
    // we
    // compute each kernel once? ->cycle through directed_gamma instead of
    // through
    // kernels
    Column eta = Column::Constant(_number_of_kernels, 1, 0.0);
    for (Index eta_index = 0; eta_index < _number_of_kernels; ++eta_index)
    {
        // compute current kernel matrix K
        _logger->trace("base kernel :\n{}", _kernels_handler[eta_index]);
        eta(eta_index) = directed_gamma.transpose() *
                         _kernels_handler[eta_index] * directed_gamma;

        _logger->trace("d(y)_r :\n{}", eta(eta_index));
    }
    _logger->trace("pre norming eta :\n{}", eta);
    eta /=
    eta.sum(); // l1-norm, as eta_i is >=0 by construction ( kernels are SDP)
    _logger->trace("eta :\n{}", eta);
    _logger->debug("compute_weights() done");
    return eta;
}

template <typename Scalar, typename Kernel>
Scalar EasyMKL<Scalar, Kernel>::compute_bias(const Column &gamma,
                                             const Column &directed_gamma,
                                             const Column &eta)
{
    return 0.5 * gamma.transpose() * (_kernels_handler.sum(eta)) * directed_gamma;
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::setup(std::vector<std::string> labels)
{
    _logger->debug("setup() start");
    _number_of_support_vectors = _kernels_handler.get_lhs_size();
    _number_of_kernels = _kernels_handler.get_number_of_kernels();
    if (_number_of_support_vectors != labels.size())
        throw std::length_error("sample size and lables size do not match");

    // manage labels
    _class_map = mimkl::data_structures::indexing_from_vector_of_strings(labels);

    std::sort(labels.begin(), labels.end()); // pass labels as value
    _unique_labels = labels;
    auto last = std::unique(_unique_labels.begin(), _unique_labels.end());
    _unique_labels.erase(last, _unique_labels.end());
    _number_of_classes = _unique_labels.size();
    _binary = _number_of_classes < 3;

    // treat the binary classification case
    _number_of_dichotomies = _binary ? 1 : _number_of_classes;
    _logger->debug("is binary : {}", _binary);
    _logger->debug("numer of dichotomies: {}", _number_of_dichotomies);

    _gammas.resize(_number_of_support_vectors, _number_of_dichotomies);
    _etas.resize(_number_of_kernels, _number_of_dichotomies);
    _biases.resize(1, _number_of_dichotomies);

    _kernels_sum = _kernels_handler.sum();
    _logger->debug("setup() done");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::train(const Index &class_index)
{
    Column y = get_dichotomy(_unique_labels[class_index]);
    _logger->debug("get_dichotomy() done");
    // gamma on plain sum
    std::pair<Column, Column> gamma_pair =
    optimize_gamma(_regularization_factor, y);
    _logger->debug("optimize_gamma() done");
    Column eta = compute_weights(gamma_pair.second);
    _logger->debug("compute_weights() done");
    // gamma on weighted sum, like KOMD on given kernel
    gamma_pair = optimize_gamma(false, y, eta);
    _logger->debug("optimize_gamma() weighted done");
    Scalar bias = compute_bias(gamma_pair.first, gamma_pair.second, eta);
    _logger->debug("compute_bias() done");

    // update state for the given dichotomy
    _gammas.col(class_index) << gamma_pair.first;
    _etas.col(class_index) << eta;
    _biases.col(class_index) << bias;
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::fit(const std::vector<std::string> &labels)
{
    _logger->debug("fit() start");
    setup(labels);
    // or iterate over class_index?
    for (Index i = 0; i < _number_of_dichotomies; ++i)
        train(i);
    _trained = true;
    _logger->debug("fit() done");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::fit(const Matrix &X,
                                  const std::vector<std::string> &labels)
{
    _logger->debug("fit() data start");
    _kernels_handler.set_lhs(X);
    fit(labels);
    _logger->debug("fit() data done");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::fit(const std::vector<Matrix> &kernel_matrices,
                                  const std::vector<std::string> &labels)
{
    _logger->debug("fit() kernel matrices start");
    _kernels_handler.set_matrices(kernel_matrices, true);
    fit(labels);
    _logger->debug("fit() kernel matrices done");
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Column EasyMKL<Scalar, Kernel>::distances(
const Column &y, const Column &gamma, const Column &eta, const Scalar &bias)
{
    _logger->debug("about to break?");
    return (_kernels_handler.sum(eta).transpose() * y.cwiseProduct(gamma)).array() -
           bias;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::decision_function()
{
    Matrix D(_kernels_handler.get_rhs_size(), _number_of_classes);
    _logger->debug("decision_function() start");
    for (Index i = 0; i < _number_of_dichotomies; ++i)
    {
        D.col(i) << distances(get_dichotomy(_unique_labels[i]), _gammas.col(i),
                              _etas.col(i), _biases(i));
    }
    if (_binary)
    {
        D.col(1) << -D.col(0);
    }
    _logger->debug("decision_function() done");
    return D;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::decision_function(const Matrix &X)
{
    if (!_trained)
        throw std::logic_error("The model should be trained first (after "
                               "instantiation or change in parameters)");
    _kernels_handler.set_rhs(X);
    return decision_function();
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::decision_function(const std::vector<Matrix> &kernel_matrices)
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
    return decision_function();
}

template <typename Scalar, typename Kernel>
std::vector<std::string>
EasyMKL<Scalar, Kernel>::predict(const std::vector<Matrix> &kernel_matrices)
{
    _logger->debug("predict() matricial start");
    Matrix D = decision_function(kernel_matrices);
    std::vector<std::string> prediction;
    prediction.reserve(_kernels_handler.get_rhs_size());
    // argmax
    typename Matrix::Index max_index;
    for (Index i = 0; i < _kernels_handler.get_rhs_size(); ++i)
    {
        D.row(i).maxCoeff(&max_index);
        _logger->trace("max_index[{}]:\t {}", i, max_index);
        prediction.push_back(_unique_labels[max_index]);
    }
    _logger->debug("predict() matricial done");
    return prediction;
}

template <typename Scalar, typename Kernel>
std::vector<std::string> EasyMKL<Scalar, Kernel>::predict(const Matrix &X)
{
    _logger->debug("predict() functional start");
    Matrix D = decision_function(X);
    std::vector<std::string> prediction;
    prediction.reserve(_kernels_handler.get_rhs_size());
    // argmax
    typename Matrix::Index max_index;
    for (Index i = 0; i < _kernels_handler.get_rhs_size(); ++i)
    {
        D.row(i).maxCoeff(&max_index);
        _logger->trace("max_index[{}]:\t {}", i, max_index);
        prediction.push_back(_unique_labels[max_index]);
    }
    _logger->debug("predict() functional done");
    return prediction;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::predict_proba(const std::vector<Matrix> &kernel_matrices)
{
    _logger->debug("predict_proba() matricial done");
    Matrix D = decision_function(kernel_matrices);
    _logger->debug("predict_proba() matricial done");
    return mimkl::linear_algebra::rowwise_soft_max(D);
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::predict_proba(const Matrix &X)
{
    _logger->debug("predict_proba() functional done");
    Matrix D = decision_function(X);
    _logger->debug("predict_proba() functional done");
    return mimkl::linear_algebra::rowwise_soft_max(D);
}

template <typename Scalar, typename Kernel>
double EasyMKL<Scalar, Kernel>::get_lambda() const
{
    return _lambda;
}

template <typename Scalar, typename Kernel>
double EasyMKL<Scalar, Kernel>::get_epsilon() const
{
    return _epsilon;
}

template <typename Scalar, typename Kernel>
bool EasyMKL<Scalar, Kernel>::get_regularization_factor() const
{
    return _regularization_factor;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix EasyMKL<Scalar, Kernel>::get_gammas() const
{
    return _gammas;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix EasyMKL<Scalar, Kernel>::get_etas() const
{
    return _etas;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Row EasyMKL<Scalar, Kernel>::get_biases() const
{
    return _biases;
}

template <typename Scalar, typename Kernel>
Indexing EasyMKL<Scalar, Kernel>::get_class_map() const
{
    return _class_map;
}

template <typename Scalar, typename Kernel>
std::vector<std::string> EasyMKL<Scalar, Kernel>::get_one_versus_rest_order() const
{
    return _unique_labels;
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::get_optimal_kernel()
{
    if (!_trained)
        throw std::logic_error(
        "The model has not been fit, kernel weights are undetermined");
    return _kernels_handler.sum(
    _etas.rowwise().mean()); // TODO treat exception for binary problem
}

template <typename Scalar, typename Kernel>
typename EasyMKL<Scalar, Kernel>::Matrix
EasyMKL<Scalar, Kernel>::get_optimal_kernel_by_class_index(const Index i)
{
    if (!_trained)
        throw std::logic_error(
        "The model has not been fit, kernel weights are undetermined");
    return _kernels_handler.sum(_etas.col(i));
}

template <typename Scalar, typename Kernel>
std::vector<typename EasyMKL<Scalar, Kernel>::Matrix>
EasyMKL<Scalar, Kernel>::get_optimal_kernels()
{
    if (!_trained)
        throw std::logic_error(
        "The model has not been fit, kernel weights are undetermined");
    std::vector<Matrix> all_optimal_kernels;
    all_optimal_kernels.reserve(_number_of_dichotomies);
    for (Index i = 0; i < _number_of_dichotomies; ++i)
        all_optimal_kernels.push_back(_kernels_handler.sum(_etas.col(i)));
    return all_optimal_kernels;
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_lambda(const double lambda)
{
    _lambda = lambda;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_epsilon(const double epsilon)
{
    _epsilon = epsilon;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_regularization_factor(
const bool regularization_factor)
{
    _regularization_factor = regularization_factor;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_parameters(const double lambda,
                                             const double epsilon,
                                             const bool regularization_factor)
{
    _lambda = lambda;
    _epsilon = epsilon;
    _regularization_factor = regularization_factor;
    _trained = false;
    _logger->debug("changing parameters requires refitting before prediction");
    _logger->debug("lambda:\n{}", _lambda);
    _logger->debug("epsilon:\n{}", _epsilon);
}

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_gammas(const Matrix &gammas)
{
    _gammas = gammas;
};

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_etas(const Matrix &etas)
{
    _etas = etas;
};

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_biases(const Row &biases)
{
    _biases = biases;
};

template <typename Scalar, typename Kernel>
void EasyMKL<Scalar, Kernel>::set_class_map(const Indexing &class_map)
{
    _class_map = class_map;
    // get companion objects
    std::vector<std::string> labels;
    labels.reserve(class_map.size());
    // keep linear complexity
    for (Indexing::const_iterator it = _class_map.begin();
         it != _class_map.end();)
    {
        const auto key = it->first;
        labels.push_back(key);
        do
        {
            ++it;
        } while (it != _class_map.end() && key == it->first);
    }
    std::sort(labels.begin(), labels.end()); // pass labels as value
    _unique_labels = labels;
    auto last = std::unique(_unique_labels.begin(), _unique_labels.end());
    _unique_labels.erase(last, _unique_labels.end());

    _number_of_classes = _unique_labels.size();
    _binary = _number_of_classes < 3;
    _number_of_dichotomies = _binary ? 1 : _number_of_classes;
};

} // namespace models
} // namespace mimkl

#endif /* INCLUDE_MIMKL_MODELS_EASY_MKL_HPP_ */
