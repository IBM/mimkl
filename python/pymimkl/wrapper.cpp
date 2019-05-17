#include <iostream>
#include <mimkl/data_structures.hpp>
#include <mimkl/definitions.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/kernels_handler.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/models.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;


using mimkl::definitions::Column;
using mimkl::definitions::Index;
using mimkl::definitions::Indexing;
using mimkl::definitions::Kernel;
using mimkl::definitions::Matrix;
using mimkl::definitions::RefConstMatrix;
using mimkl::definitions::ReversedIndexing;
using mimkl::definitions::Row;
using mimkl::definitions::SparseMatrix;

typedef mimkl::kernels_handler::KernelsHandler<double, Kernel> _KernelsHandler;
typedef mimkl::models::EasyMKL<double, Kernel> _EasyMKL;
typedef mimkl::models::UMKLKNN<double, Kernel> _UMKLKNN;
typedef mimkl::models::AverageMKL<double, Kernel> _AverageMKL;
typedef mimkl::solvers::KOMD<double> _KOMD;

py::return_value_policy kernel_return_policy = py::return_value_policy::move;

void log_level_helper(const std::string logger_name,
                      const spdlog::level::level_enum level)
{
    std::shared_ptr<spdlog::logger> logger = spdlog::get(logger_name);
    if (logger != NULL)
    {
        logger->set_level(level);
    }
    else
    {
        throw std::invalid_argument("this logger was not found");
    }
}
auto console = spdlog::stdout_color_mt("console");

PYBIND11_MODULE(_pymimkl, m)
{
    m.doc() = "matrix induced multiple kernel learning module written in C++ "
              "wrapped by pybind11";

    // models
    // KernelsHandler
    py::class_<_KernelsHandler>(m, "KernelsHandler")
    .def(py::init<const std::vector<Kernel> &, const bool, const bool>(),
         "kernel_functions"_a, "precompute"_a = true,
         "trace_normalization"_a = true)
    .def(py::init<const std::vector<Matrix> &, const bool, const bool>(),
         "kernel_matrices"_a, "precompute"_a = true,
         "trace_normalization"_a = true)
    .def("set_lhs", &_KernelsHandler::set_lhs, "X"_a)
    .def("set_rhs", &_KernelsHandler::set_rhs, "X"_a)
    .def("set_matrices", &_KernelsHandler::set_matrices, "kernel_matrices"_a,
         "learning_mode"_a)
    .def("set_functions", &_KernelsHandler::set_functions, "kernel_functions"_a,
         "learning_mode"_a)
    .def_property_readonly("lhs", &_KernelsHandler::get_lhs)
    .def_property_readonly("precompute", &_KernelsHandler::get_precompute)
    .def_property_readonly("trace_normalization",
                           &_KernelsHandler::get_trace_normalization)
    .def_property_readonly("kernels_size", &_KernelsHandler::get_number_of_kernels)
    .def_property_readonly("lhs_size", &_KernelsHandler::get_lhs_size)
    .def_property_readonly("rhs_size", &_KernelsHandler::get_rhs_size)
    .def_property_readonly("kernel_matrices", &_KernelsHandler::get_matrices)
    .def_property_readonly("kernel_functions", &_KernelsHandler::get_functions)
    .def_property_readonly("trace_factors", &_KernelsHandler::get_trace_factors)
    .def("get_corrected_kernels_weights",
         &_KernelsHandler::get_corrected_kernels_weights, "kernels_weights"_a,
         "get kernels weights to use for original, untracenormalized kernels, "
         "so weighted sum is learned kernel (impacts interpretation of "
         "weights)")
    .def("sum", py::overload_cast<>(&_KernelsHandler::sum))
    .def("sum", py::overload_cast<const COLUMN(double) &>(&_KernelsHandler::sum))
    .def("__getitem__",
         &_KernelsHandler::operator[]); // only access, not:
                                        // assignment,slicing,...

    // EasyMKL
    py::class_<_EasyMKL>(m, "EasyMKL_")
    .def(py::init<const std::vector<Kernel> &, const bool, const bool,
                  const double, const double, const bool>(),
         "kernel_functions"_a = std::vector<Kernel>(), "precompute"_a = true,
         "trace_normalization"_a = true, "lam"_a = 0.8, "epsilon"_a = 0.0001,
         "regularization_factor"_a = false)
    .def(py::init<const std::vector<Matrix> &, const bool, const bool,
                  const double, const double, const bool>(),
         "kernel_matrices"_a = std::vector<Matrix>(), "precompute"_a = true,
         "trace_normalization"_a = true, "lam"_a = 0.8, "epsilon"_a = 0.0001,
         "regularization_factor"_a = false)
    .def("fit",
         py::overload_cast<const Matrix &, const std::vector<std::string> &>(
         &_EasyMKL::fit),
         "X"_a, "labels"_a)
    .def("fit",
         py::overload_cast<const std::vector<Matrix> &,
                           const std::vector<std::string> &>(&_EasyMKL::fit),
         "kernel_matrices"_a, "labels"_a)
    .def("predict", py::overload_cast<const Matrix &>(&_EasyMKL::predict), "X"_a)
    .def("predict",
         py::overload_cast<const std::vector<Matrix> &>(&_EasyMKL::predict),
         "kernel_matrices"_a)
    .def("predict_proba",
         py::overload_cast<const Matrix &>(&_EasyMKL::predict_proba), "X"_a)
    .def("predict_proba",
         py::overload_cast<const std::vector<Matrix> &>(&_EasyMKL::predict_proba),
         "kernel_matrices"_a)
    .def("decision_function",
         py::overload_cast<const Matrix &>(&_EasyMKL::decision_function), "X"_a)
    .def("decision_function",
         py::overload_cast<const std::vector<Matrix> &>(
         &_EasyMKL::decision_function),
         "kernel_matrices"_a)
    .def("get_one_versus_rest_order", &_EasyMKL::get_one_versus_rest_order,
         "return the set of ordered class labels for all 1 versus rest "
         "problems")
    .def("get_optimal_kernel", py::overload_cast<>(&_EasyMKL::get_optimal_kernel),
         "return the learned kernel, the sum is weigthed averaging over all "
         "1 versus rest problems weights")
    .def("get_optimal_kernel_by_class_index",
         py::overload_cast<const Index>(
         &_EasyMKL::get_optimal_kernel_by_class_index),
         "return the learned kernel of the indicated 1 versus rest problem")
    .def("get_optimal_kernels", &_EasyMKL::get_optimal_kernels,
         "return all the learned kernels for each 1 versus rest problem")
    .def_property_readonly("support_vectors", &_EasyMKL::get_support_vectors)
    .def_property_readonly("biases", &_EasyMKL::get_biases)
    .def_property_readonly("etas", &_EasyMKL::get_etas)
    .def_property_readonly("gammas", &_EasyMKL::get_gammas)
    .def_property_readonly("kernel_matrices", &_EasyMKL::get_matrices)
    .def_property_readonly("kernel_functions", &_EasyMKL::get_functions)
    .def_property_readonly("trace_factors", &_EasyMKL::get_trace_factors)
    .def("set_parameters", &_EasyMKL::set_parameters, "lam"_a = 1,
         "epsilon"_a = 0.001, "regularization_factor"_a = false,
         "set regularization parameter lam and epsilon for optimization "
         "termination ")
    .def_property("lam", &_EasyMKL::get_lambda, &_EasyMKL::set_lambda)
    .def_property("epsilon", &_EasyMKL::get_epsilon, &_EasyMKL::set_epsilon)
    .def_property("regularization_factor", &_EasyMKL::get_regularization_factor,
                  &_EasyMKL::set_regularization_factor)
    .def("get_corrected_kernels_weights",
         &_EasyMKL::get_corrected_kernels_weights, "kernels_weights"_a,
         "get kernels weights to use for original, untracenormalized kernels, "
         "so weighted sum is learned kernel (impacts interpretation of "
         "weights)")
    .def(py::pickle(
    [](const _EasyMKL &instance) { // __getstate__
        // make _class_map pickable
        ReversedIndexing reversed_class_map;
        for (const auto &entry : instance.get_class_map())
        {
            reversed_class_map.emplace(entry.second, entry.first);
        }
        // encode the instance
        return py::make_tuple(instance.get_functions(), instance.get_precompute(),
                              instance.get_trace_normalization(),
                              instance.get_lambda(), instance.get_epsilon(),
                              instance.get_regularization_factor(),
                              instance.get_gammas(), instance.get_etas(),
                              instance.get_biases(), reversed_class_map,
                              instance.get_support_vectors());
    },
    [](py::tuple t) { // __setstate__
        if (t.size() != 11)
            throw std::runtime_error("Invalid state for EasyMKL object!");

        // Create a new instance
        _EasyMKL instance(t[0].cast<std::vector<Kernel>>(), t[1].cast<bool>(),
                          t[2].cast<bool>(), t[3].cast<double>(),
                          t[4].cast<double>(), t[5].cast<bool>());

        // Assign any additional state
        instance.set_gammas(t[6].cast<Matrix>());
        instance.set_etas(t[7].cast<Matrix>());
        instance.set_biases(t[8].cast<Row>());
        // Create Indexing representing the class_map
        Indexing class_map;
        for (const auto &entry : t[9].cast<ReversedIndexing>())
        {
            class_map.emplace(entry.second, entry.first);
        }
        instance.set_class_map(class_map);
        instance.set_support_vectors(t[10].cast<Matrix>());

        return instance;
    }));

    // UMKLKNN
    py::class_<_UMKLKNN>(m, "UMKLKNN_")
    .def(py::init<const std::vector<Kernel> &, const bool, const bool,
                  const Index, const double, const Index>(),
         "kernel_functions"_a = std::vector<Kernel>(), "precompute"_a = true,
         "trace_normalization"_a = true, "k"_a = 5, "epsilon"_a = 0.0001,
         "maxiter_qp"_a = 100000)
    .def(py::init<const std::vector<Matrix> &, const bool, const bool,
                  const Index, const double, const Index>(),
         "kernel_matrices"_a = std::vector<Matrix>(), "precompute"_a = true,
         "trace_normalization"_a = true, "k"_a = 5, "epsilon"_a = 0.0001,
         "maxiter_qp"_a = 100000)
    .def("fit", py::overload_cast<const Matrix &>(&_UMKLKNN::fit), "X"_a,
         "provide X_train to the kernels_handler and fit()")
    .def("fit", py::overload_cast<const std::vector<Matrix> &>(&_UMKLKNN::fit),
         "kernel_matrices"_a,
         "provide kernel_matrices to the kernels_handler and fit()")
    .def("predict", py::overload_cast<const Matrix &>(&_UMKLKNN::predict), "X"_a)
    .def("predict",
         py::overload_cast<const std::vector<Matrix> &>(&_UMKLKNN::predict),
         "kernel_matrices"_a)
    .def("get_optimal_kernel", &_UMKLKNN::get_optimal_kernel,
         "return the learned kernel, that is the weighted sum of kernels")
    .def_property_readonly("support_vectors", &_UMKLKNN::get_support_vectors)
    .def_property_readonly("beta", &_UMKLKNN::get_beta)
    .def_property_readonly("kernel_matrices", &_UMKLKNN::get_matrices)
    .def_property_readonly("kernel_functions", &_UMKLKNN::get_functions)
    .def_property_readonly("trace_factors", &_UMKLKNN::get_trace_factors)
    .def("set_parameters", &_UMKLKNN::set_parameters, "k"_a,
         "epsilon"_a = 0.001, "maxiter_qp"_a = 10000,
         "set number of nearest neighbors k and epsilon for optimization "
         "termination ")
    .def_property("k", &_UMKLKNN::get_k, &_UMKLKNN::set_k)
    .def_property("epsilon", &_UMKLKNN::get_epsilon, &_UMKLKNN::set_epsilon)
    .def_property("maxiter_qp", &_UMKLKNN::get_maxiter_qp,
                  &_UMKLKNN::set_maxiter_qp)
    .def("get_corrected_kernels_weights",
         &_UMKLKNN::get_corrected_kernels_weights, "kernels_weights"_a,
         "get kernels weights to use for original, untracenormalized kernels, "
         "so weighted sum is learned kernel (impacts interpretation of "
         "weights)")
    .def(py::pickle(
    [](const _UMKLKNN &instance) { // __getstate__
        // encode the instance
        return py::make_tuple(instance.get_functions(), instance.get_precompute(),
                              instance.get_trace_normalization(),
                              instance.get_k(), instance.get_epsilon(),
                              instance.get_maxiter_qp(), instance.get_beta(),
                              instance.get_support_vectors());
    },
    [](py::tuple t) { // __setstate__
        if (t.size() != 8)
            throw std::runtime_error("Invalid state for UMKLKNN object!");

        // Create a new instance
        _UMKLKNN instance(t[0].cast<std::vector<Kernel>>(), t[1].cast<bool>(),
                          t[2].cast<bool>(), t[3].cast<Index>(),
                          t[4].cast<double>(), t[5].cast<Index>());

        // Assign any additional state
        instance.set_beta(t[6].cast<Column>());
        instance.set_support_vectors(t[7].cast<Matrix>());

        return instance;
    }));


    // Average
    py::class_<_AverageMKL>(m, "AverageMKL_")
    .def(py::init<const std::vector<Kernel> &, const bool, const bool>(),
         "kernel_functions"_a = std::vector<Kernel>(), "precompute"_a = true,
         "trace_normalization"_a = true)
    .def(py::init<const std::vector<Matrix> &, const bool, const bool>(),
         "kernel_matrices"_a = std::vector<Matrix>(), "precompute"_a = true,
         "trace_normalization"_a = true)
    .def("fit", py::overload_cast<const Matrix &>(&_AverageMKL::fit), "X"_a,
         "provide X_train to the kernels_handler and fit()")
    .def("fit", py::overload_cast<const std::vector<Matrix> &>(&_AverageMKL::fit),
         "kernel_matrices"_a,
         "provide kernel_matrices to the kernels_handler and fit()")
    .def("predict", py::overload_cast<const Matrix &>(&_AverageMKL::predict),
         "X"_a)
    .def("predict",
         py::overload_cast<const std::vector<Matrix> &>(&_AverageMKL::predict),
         "kernel_matrices"_a)
    .def("get_optimal_kernel", &_AverageMKL::get_optimal_kernel,
         "return the learned kernel, that is the weighted sum of kernels")
    .def_property_readonly("support_vectors", &_AverageMKL::get_support_vectors)
    .def_property_readonly("kernel_matrices", &_AverageMKL::get_matrices)
    .def_property_readonly("kernel_functions", &_AverageMKL::get_functions)
    .def_property_readonly("trace_factors", &_AverageMKL::get_trace_factors)
    .def_property("weights", &_AverageMKL::get_weights, &_AverageMKL::set_weights)
    .def("get_corrected_kernels_weights",
         &_AverageMKL::get_corrected_kernels_weights, "kernels_weights"_a,
         "get kernels weights to use for original, untracenormalized kernels, "
         "so weighted sum is learned kernel (impacts interpretation of "
         "weights)")
    .def(py::pickle(
    [](const _AverageMKL &instance) { // __getstate__
        // encode the instance
        return py::make_tuple(instance.get_functions(), instance.get_precompute(),
                              instance.get_trace_normalization(),
                              instance.get_weights(),
                              instance.get_support_vectors());
    },
    [](py::tuple t) { // __setstate__
        if (t.size() != 5)
            throw std::runtime_error("Invalid state for AverageMKL object!");

        // Create a new instance
        _AverageMKL instance(t[0].cast<std::vector<Kernel>>(),
                             t[1].cast<bool>(), t[2].cast<bool>());

        // Assign any additional state
        instance.set_weights(t[3].cast<Column>());
        instance.set_support_vectors(t[4].cast<Matrix>());

        return instance;
    }));

    // KOMD
    py::class_<_KOMD>(m, "KOMD")
    .def(py::init<const Matrix &, const double, const double, const bool>(),
         "K"_a, "lam"_a = 0, "epsilon"_a = 0.001,
         "regularization_factor"_a = false)
    .def("solve", &_KOMD::solve, "y"_a)
    .def_property_readonly("gamma", &_KOMD::get_result)
    .def("set_parameters", &_KOMD::set_parameters, "lam"_a = 1,
         "epsilon"_a = 0.001,
         "set regularization parameter lambda and "
         "epsilon for optimization termination ")
    .def_property("lam", &_KOMD::get_lambda, &_KOMD::set_lambda)
    .def_property("epsilon", &_KOMD::get_epsilon, &_KOMD::set_epsilon);

    // LogLevel
    py::enum_<spdlog::level::level_enum>(m, "Loglevel")
    .value("trace", spdlog::level::trace)
    .value("debug", spdlog::level::debug)
    .value("info", spdlog::level::info)
    .value("warn", spdlog::level::warn)
    .value("err", spdlog::level::err)
    .value("critical", spdlog::level::critical)
    .value("off", spdlog::level::off)
    .export_values();

    // functions
    m.def("set_global_level", &spdlog::set_level, "level"_a,
          "set the global verbosity, default is Loglevel.info");
    m.def("set_level", &log_level_helper, "logger"_a, "level"_a,
          "set the verbosity, default is global Loglevel");
    m.def("dichotomies", &mimkl::data_structures::dichotomies<double>,
          "labels"_a);
    m.def("aggregate_weighted_kernels",
          &mimkl::linear_algebra::aggregate_weighted_kernels<double, Kernel>,
          "X"_a, "Y"_a, "functions"_a, "weights"_a, kernel_return_policy);
    m.def("normalize_kernel",
          &mimkl::linear_algebra::normalize_kernel<RefConstMatrix>, "K"_a);
    m.def("normalize_kernel_prediction",
          &mimkl::linear_algebra::normalize_kernel_prediction<
          RefConstMatrix, RefConstMatrix, RefConstMatrix>,
          "K"_a, "X"_a, "Y"_a);
    m.def("centralize_kernel",
          &mimkl::linear_algebra::centralize_kernel<RefConstMatrix>, "K"_a);
    // kernels
    m.def("linear_kernel",
          &mimkl::kernel::linear_kernel<Matrix, RefConstMatrix, RefConstMatrix>,
          "X"_a, "Y"_a, kernel_return_policy);
    m.def("polynomial_kernel",
          &mimkl::kernel::polynomial_kernel<Matrix, RefConstMatrix, RefConstMatrix>,
          "X"_a, "Y"_a, "degree"_a, "offset"_a, kernel_return_policy);
    m.def("gaussian_kernel",
          &mimkl::kernel::gaussian_kernel<Matrix, RefConstMatrix, RefConstMatrix>,
          "X"_a, "Y"_a, "sigma_square"_a, kernel_return_policy);
    m.def("sigmoidal_kernel",
          &mimkl::kernel::sigmoidal_kernel<Matrix, RefConstMatrix, RefConstMatrix>,
          "X"_a, "Y"_a, "a"_a, "b"_a, kernel_return_policy);
    // inductions
    m.def("induce_linear_kernel",
          &mimkl::induction::induce_linear_kernel<Matrix, RefConstMatrix,
                                                  RefConstMatrix, SparseMatrix>,
          "X"_a, "Y"_a, "L"_a, kernel_return_policy);
    m.def("induce_polynomial_kernel",
          &mimkl::induction::induce_polynomial_kernel<Matrix, RefConstMatrix,
                                                      RefConstMatrix, SparseMatrix>,
          "X"_a, "Y"_a, "L"_a, "degree"_a, "offset"_a, kernel_return_policy);
    m.def("induce_gaussian_kernel",
          &mimkl::induction::induce_gaussian_kernel<Matrix, RefConstMatrix,
                                                    RefConstMatrix, SparseMatrix>,
          "X"_a, "Y"_a, "L"_a, "sigma_square"_a, kernel_return_policy);
    m.def("induce_sigmoidal_kernel",
          &mimkl::induction::induce_sigmoidal_kernel<Matrix, RefConstMatrix,
                                                     RefConstMatrix, SparseMatrix>,
          "X"_a, "Y"_a, "L"_a, "a"_a, "b"_a, kernel_return_policy);
}
