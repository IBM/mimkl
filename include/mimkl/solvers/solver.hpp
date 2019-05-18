#ifndef INCLUDE_MIMKL_SOLVERS_SOLVER_HPP_
#define INCLUDE_MIMKL_SOLVERS_SOLVER_HPP_

#include <Eigen/Core>
#include <dlib/optimization.h>
#include <iostream>
#include <mimkl/definitions.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

using dlib::mat;

namespace mimkl
{
namespace solvers
{

template <typename Scalar>
class Solver
{
    private:
    typedef COLUMN(Scalar) Column;
    typedef Eigen::Map<Column> MapColumn;
    virtual void solve(const Column &) = 0;

    public:
    // the setup/construction depends heavily on the child
    //  virtual MapColumn get_result() const = 0;

    virtual ~Solver() = default;
};

} // namespace solvers
} // namespace mimkl
#endif /* INCLUDE_MIMKL_SOLVERS_SOLVER_HPP_ */
