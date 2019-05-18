#ifndef INCLUDE_MIMKL_SOLVERS_KOMD_HPP_
#define INCLUDE_MIMKL_SOLVERS_KOMD_HPP_

#include <Eigen/Core>
#include <dlib/optimization.h>
#include <iostream>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <mimkl/solvers/solver.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using dlib::mat;

namespace mimkl
{
namespace solvers
{

template <typename Scalar>
class KOMD : private Solver<Scalar>
{

    private:
    // logger instance
    static std::shared_ptr<spdlog::logger> _logger;
    // typedefs
    typedef MATRIX(Scalar) Matrix;
    typedef COLUMN(Scalar) Column;
    typedef Eigen::Map<const Column> MapColumn;
    // ensure Col-Major for dlib (default is row-major) as we use Eigen::Map<>
    // on
    // it.
    typedef dlib::matrix<Scalar,
                         0,
                         1,
                         dlib::default_memory_manager,
                         dlib::column_major_layout>
    DlibColumn; // runtime sized
                // column
                // vector.  -->
                // LibColumn a ;
                // a.set_size()
    // members
    // trade-off hard-svm vs centroids 0<=lambda<=1
    double _lambda;
    double _epsilon;
    bool _regularization_factor;
    Scalar _reg_factor = 1;
    dlib::solve_qp2_using_smo<DlibColumn> _solver;
    Matrix _K;
    //  Column _y; //boolian labels
    Matrix _Q;
    DlibColumn _gamma;
    void setup(const Column &);

    public:
    KOMD() {}
    //! Setup KOMD parameters prior to solving the QP.
    /*!
     \param K unweighted sum of (all multiple) kernels (or a single kernel)
     \param lambda trade-off parameter between hard SVM (default) vs L2
     regularization, 0<=lambda<=1.
     \param epsilon.
     */
    KOMD(const Matrix &,
         const double lambda = 0.,
         const double epsilon = 0.0001,
         const bool regularization_factor = false);

    //! Solve KOMD QP following setup.
    /*!
     \param y class labels, where y_i == 1 or y_i == -1.
     */
    void solve(const Column &);

    void set_parameters(const double, const double epsilon = 0.0001);
    void set_lambda(const double);
    void set_epsilon(const double);
    double get_lambda() const;
    double get_epsilon() const;

    // Get result from KOMD QP.
    MapColumn get_result() const;
};

template <typename Scalar>
std::shared_ptr<spdlog::logger>
KOMD<Scalar>::_logger = spdlog::stdout_color_mt("komd_console");

template <typename Scalar>
KOMD<Scalar>::KOMD(const Matrix &K,
                   const double lambda,
                   const double epsilon,
                   const bool regularization_factor)
{
    _K = K;
    _logger->debug("norm of K\n{}", K.norm());
    _epsilon = epsilon;
    _lambda = lambda;
    _regularization_factor = regularization_factor;
}

// implementation
template <typename Scalar>
void KOMD<Scalar>::setup(const Column &y)
{

    _Q = (1. - _lambda) * y.asDiagonal() * _K * y.asDiagonal(); // KLL
    _logger->debug("norm KLL:\n{}", _Q.norm());
    if (_regularization_factor)
    {
        Scalar nneg = (y.array() < 0.).count();
        Scalar npos = (y.array() > 0.).count();
        _reg_factor = nneg * npos / (nneg + npos);
        _Q.diagonal().array() += _reg_factor * _lambda; // LID
    }
    else
    {
        _Q.diagonal().array() += _lambda; // LID
    }
    _logger->debug("norm( KLL + LID):\n{}", _Q.norm());
    _Q *= 2.; // qp solves 0.5*Q
}

template <typename Scalar>
void KOMD<Scalar>::solve(const Column &y)
{
    setup(y);
    try
    {
        _solver(mat(_Q), mat(y), (double)2. / y.size(), _gamma,
                _epsilon); // FIXME wrong solver?
        _logger->trace("Q:\n{}\ny:\n{}\nnu:\n{}\ngamma "
                       "(dlib):\n{}\nepsilon:\n{}",
                       _Q, y, (double)2. / y.size(), _gamma, _epsilon);
    }
    catch (dlib::invalid_nu_error &exception)
    {
        DlibColumn dlib_y = mat(y);
        long positive_count = 0;
        long negative_count = 0;
        for (long r = 0; r < dlib_y.nr(); ++r)
        {
            if (dlib_y(r) == 1.0)
                ++positive_count;
            else if (dlib_y(r) == -1.0)
                ++negative_count;
        }
        if (positive_count == 1 || negative_count == 1)
        {
            _logger->error("y:\n{}", y);
            throw std::invalid_argument(
            "Each class in y should have more than one member");
        }
        else
            throw exception;
    }
    _logger->debug("sum over gamma: \n{}", dlib::sum(_gamma));
    _logger->debug("sum over directed gamma: \n{}",
                   dlib::sum(dlib::trans(mat(y)) * _gamma));
}

template <typename Scalar>
typename KOMD<Scalar>::MapColumn KOMD<Scalar>::get_result() const
{
    // Map Dlib to Eigen
    _logger->debug("sum over gamma (EigenMap): \n{}",
                   mimkl::linear_algebra::dlib_to_eigen<Scalar>(_gamma).sum());
    return mimkl::linear_algebra::dlib_to_eigen<Scalar>(_gamma);
}

template <typename Scalar>
void KOMD<Scalar>::set_parameters(const double lambda, const double epsilon)
{
    _epsilon = epsilon;
    _lambda = lambda;
}

template <typename Scalar>
double KOMD<Scalar>::get_lambda() const
{
    return _lambda;
}

template <typename Scalar>
void KOMD<Scalar>::set_lambda(const double lambda)
{
    _lambda = lambda;
}

template <typename Scalar>
double KOMD<Scalar>::get_epsilon() const
{
    return _epsilon;
}

template <typename Scalar>
void KOMD<Scalar>::set_epsilon(const double epsilon)
{
    _epsilon = epsilon;
}

} // namespace solvers
} // namespace mimkl

#endif /* #ifndef INCLUDE_MIMKL_SOLVERS_KOMD_HPP_ */
