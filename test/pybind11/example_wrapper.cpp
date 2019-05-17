#include "example.hpp"
#include <mimkl/definitions.hpp>
#include <mimkl/kernels.hpp>
#include <mimkl/linear_algebra.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

using mimkl::definitions::Kernel;
using mimkl::definitions::Matrix;
using mimkl::definitions::RefConstMatrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

py::return_value_policy kernel_return_policy = py::return_value_policy::move;

PYBIND11_MODULE(example, m)
{
    // define the class
    py::class_<Example> example(m, "Example");
    example.def(py::init<>())
    .def("set", &Example::set)
    .def("greet", &Example::greet)
    .def("many", &Example::many);

    m.def("add_to_3", &add_to_3);
    m.def("apply_5", &apply_5);
    m.def("map1", &map);
    m.def("zip_map", &zip_map);

    m.def("a_mat", []() { return Eigen::MatrixXi::Constant(3, 3, 42); });

    m.def("add_sparse", &add_sparse);

    m.def("sum_2_mats",
          [](const Eigen::MatrixXi a,
             const Eigen::MatrixXi b) -> Eigen::MatrixXi { return a + b; });

    m.def("linear_kernel_cpp",
          &mimkl::kernel::linear_kernel<Matrix, RefConstMatrix, RefConstMatrix>,
          "X"_a, "Y"_a, kernel_return_policy);

    m.def("sum_plain_matrices", &sum_plain_matrices);
    m.def("sum_plain_matrices_lin_alg",
          &mimkl::linear_algebra::sum_matrices<double>);

    m.def("sum_matrices", &sum_matrices);

    m.def("sum_matrices_temp", &sum_matrices_temp<double, nonvoid_fun>);
    m.def("sum_matrices_temp_inline",
          &sum_matrices_temp_inline<double, nonvoid_fun>);

    m.def("sum_temp", &sum_temp<double, std::function<double(double, double)>>);
    m.def(
    "sum_mat_return_scalar_temp",
    &sum_mat_return_scalar_temp<double, std::function<double(Matrix, Matrix)>>);

}
