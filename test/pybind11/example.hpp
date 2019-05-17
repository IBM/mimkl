#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <mimkl/definitions.hpp>
#include <mimkl/linear_algebra.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using mimkl::definitions::Index;
typedef std::function<
void(const MATRIX(double) &, const MATRIX(double) &, MATRIX(double) &)>
void_fun;
typedef std::function<MATRIX(double)(const MATRIX(double) &, const MATRIX(double) &)>
nonvoid_fun;

class Example
{
    public:
    void set(std::string msg) { _msg = msg; }
    void many(const std::vector<std::string> &msgs);
    std::string greet() { return _msg; }

    private:
    std::string _msg;
};

double apply_5(std::function<double(double)>);
double add_to_3(double);
std::vector<double> map(std::function<double(double)>, std::vector<double>);
std::vector<double>
zip_map(std::vector<std::function<double(double)>>, std::vector<double>);
MATRIX(double) sum_plain_matrices(const std::vector<MATRIX(double)>);
MATRIX(double)
sum_matrices(const MATRIX(double) &,
             const MATRIX(double) &,
             const std::vector<nonvoid_fun> &);

template <typename Scalar, typename Function>
MATRIX(Scalar)
sum_matrices_temp(const MATRIX(Scalar) & lhs,
                  const MATRIX(Scalar) & rhs,
                  const std::vector<Function> &Ks)
{
    // fold over Ks
    return std::accumulate(Ks.begin(), Ks.end(),
                           MATRIX(Scalar)::Zero(lhs.rows(), rhs.rows()).eval(),
                           [&, lhs, rhs](const MATRIX(Scalar) & acc,
                                         const Function &fun) {
                               return acc + fun(lhs, rhs);
                           });
}

template <typename Scalar, typename Function>
MATRIX(Scalar)
sum_matrices_temp_inline(const MATRIX(Scalar) & lhs,
                         const MATRIX(Scalar) & rhs,
                         const std::vector<Function> &Ks)
{
    MATRIX(Scalar)
    acc = MATRIX(Scalar)::Zero(lhs.rows(), rhs.rows()).eval();
    // for( auto iter = Ks.begin(); iter != Ks.end(); ++iter){
    for (auto iter : Ks)
    {
        // acc = acc + (*iter)(lhs,rhs);
        acc += iter(lhs, rhs); // (*iter)
    }
    return acc;
}

template <typename Scalar, typename Function>
Scalar sum_temp(const Scalar &x, const Scalar &y, const std::vector<Function> &fs)
{
    // fold over Ks
    return std::accumulate(fs.begin(), fs.end(), (Scalar)0,
                           [&](const Scalar &acc, const Function &fun) {
                               return acc + fun(x, y);
                           });
}

template <typename Scalar, typename Function>
Scalar sum_mat_return_scalar_temp(const MATRIX(Scalar) & x,
                                  const MATRIX(Scalar) & y,
                                  const std::vector<Function> &fs)
{
    // fold over Ks
    return std::accumulate(fs.begin(), fs.end(), (Scalar)0,
                           [&](const Scalar &acc, const Function &fun) {
                               return acc + fun(x, y);
                           });
}

MATRIX(double)
sum_fun_matrices_reference(const MATRIX(double) &,
                           const MATRIX(double) &,
                           const std::vector<void_fun>);
void_fun convert_function(nonvoid_fun);
nonvoid_fun convert_function(void_fun);
std::vector<void_fun> convert_function(std::vector<nonvoid_fun>);
std::vector<nonvoid_fun> convert_function(std::vector<void_fun>);
MATRIX(double)
sum_fun_matrices_return(const MATRIX(double) &,
                        const MATRIX(double) &,
                        const std::vector<nonvoid_fun>);
MATRIX(double)
sum_converted_fun_matrices_return(const MATRIX(double) &,
                                  const MATRIX(double) &,
                                  const std::vector<void_fun>);

Eigen::SparseMatrix<double>
add_sparse(Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>);
