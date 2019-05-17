#include "example.hpp"

void Example::many(const std::vector<std::string> &msgs)
{
    long l = msgs.size();
    std::stringstream ss;
    for (long i = 0; i < l; ++i)
    {
        if (i > 0)
            ss << ", ";
        std::string s = msgs[i];
        ss << s;
    }
    _msg = ss.str();
}

double apply_5(std::function<double(double)> foo) { return foo(5); }

double add_to_3(double a) { return 3 + a; }

std::vector<double> map(std::function<double(double)> f, std::vector<double> xs)
{
    for (Index i = 0; i < xs.size(); ++i)
    {
        xs[i] = f(xs[i]);
    }
    return xs;
}

std::vector<double>
zip_map(std::vector<std::function<double(double)>> fs, std::vector<double> xs)
{
    for (Index i = 0; i < xs.size(); ++i)
    {
        xs[i] = fs[i](xs[i]);
    }
    return xs;
}

MATRIX(double)
sum_plain_matrices(const std::vector<MATRIX(double)> Ks)
{
    // fold over Ks
    return std::accumulate(Ks.begin(), Ks.end(),
                           MATRIX(double)::Zero((*Ks.begin()).rows(),
                                                (*Ks.begin()).cols())
                           .eval()); //, [](a,b){operator+(a,b)}
}

MATRIX(double) // like specialized sum_matrices<Scalar>
sum_matrices(const MATRIX(double) & lhs,
             const MATRIX(double) & rhs,
             const std::vector<nonvoid_fun> &Ks)
{ // or w/o &
    // fold over Ks
    return std::accumulate(Ks.begin(), Ks.end(),
                           MATRIX(double)::Zero(lhs.rows(), rhs.rows()).eval(),
                           [&](const MATRIX(double) & acc, const nonvoid_fun &fun) {
                               return acc + fun(lhs, rhs);
                           });
}

MATRIX(double) // specialized sum_matrices<Scalar>
sum_fun_matrices_reference(const MATRIX(double) & lhs,
                           const MATRIX(double) & rhs,
                           const std::vector<void_fun> Ks)
{
    MATRIX(double) K(lhs.rows(), rhs.rows());
    // fold over Ks
    return std::accumulate(Ks.begin(), Ks.end(),
                           MATRIX(double)::Zero(lhs.rows(), rhs.rows()).eval(),
                           [&](const MATRIX(double) acc, const void_fun fun) {
                               fun(lhs, rhs, K);
                               return acc + K;
                           });
}

void_fun convert_function(nonvoid_fun f_return)
{
    void_fun f_reference =
    [f_return](const MATRIX(double) & lhs, const MATRIX(double) & rhs,
               MATRIX(double) & reference) { reference = f_return(lhs, rhs); };
    return f_reference;
}

nonvoid_fun convert_function(void_fun f_reference)
{
    nonvoid_fun f_return = [f_reference](const MATRIX(double) & lhs,
                                         const MATRIX(double) & rhs) {
        MATRIX(double) reference;
        f_reference(lhs, rhs, reference);
        return reference;
    };
    return f_return;
}

std::vector<void_fun> convert_functions(std::vector<nonvoid_fun> fs_return)
{
    std::vector<void_fun> fs_reference(fs_return.size());
    for (Index i = 0; i < fs_return.size(); ++i)
    {
        fs_reference[i] = convert_function(fs_return[i]);
    }
    return fs_reference;
}

std::vector<nonvoid_fun> convert_functions(std::vector<void_fun> fs_reference)
{
    std::vector<nonvoid_fun> fs_return(fs_reference.size());
    for (Index i = 0; i < fs_return.size(); ++i)
    {
        fs_return[i] = convert_function(fs_reference[i]);
    }
    return fs_return;
}

MATRIX(double)
sum_fun_matrices_return(const MATRIX(double) & lhs,
                        const MATRIX(double) & rhs,
                        const std::vector<nonvoid_fun> fs_return)
{
    return mimkl::linear_algebra::aggregate_kernels(lhs, rhs, fs_return);
}
MATRIX(double)
sum_converted_fun_matrices_return(const MATRIX(double) & lhs,
                                  const MATRIX(double) & rhs,
                                  const std::vector<void_fun> fs_reference)
{
    return sum_fun_matrices_return(lhs, rhs, convert_functions(fs_reference));
}

Eigen::SparseMatrix<double>
add_sparse(Eigen::SparseMatrix<double> a, Eigen::SparseMatrix<double> b)
{
    return a + b;
}

int main(int argc, char **argv) {}
