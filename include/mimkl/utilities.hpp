#ifndef INCLUDE_MIMKL_UTILITIES_HPP_
#define INCLUDE_MIMKL_UTILITIES_HPP_

#include <cxxabi.h>
#include <memory>
#include <mimkl/definitions.hpp>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

using mimkl::definitions::Index;

namespace mimkl
{
namespace utilities
{

//// print type

//  https://stackoverflow.com/a/23267260/5955544
std::string demangled(std::string const &sym);

template <class T>
std::string print_type()
{
    bool is_lvalue_reference = std::is_lvalue_reference<T>::value;
    bool is_rvalue_reference = std::is_rvalue_reference<T>::value;
    bool is_const = std::is_const<typename std::remove_reference<T>::type>::value;

    std::stringstream ss;
    ss << demangled(typeid(T).name());
    if (is_const)
    {
        ss << " const";
    }
    if (is_lvalue_reference)
    {
        ss << " &";
    }
    if (is_rvalue_reference)
    {
        ss << " &&";
    }
    ss << std::endl;

    return ss.str();
};

template <class T>
std::string print_type(const T a)
{
    bool is_lvalue_reference = std::is_lvalue_reference<T>::value;
    bool is_rvalue_reference = std::is_rvalue_reference<T>::value;
    bool is_const = std::is_const<typename std::remove_reference<T>::type>::value;

    std::stringstream ss;
    ss << demangled(typeid(T).name());
    if (is_const)
    {
        ss << " const";
    }
    if (is_lvalue_reference)
    {
        ss << " &";
    }
    if (is_rvalue_reference)
    {
        ss << " &&";
    }
    ss << std::endl;

    return ss.str();
};

// adapted from https://stackoverflow.com/a/257382/5955544
// and https://stackoverflow.com/a/982941/5955544
// and https://stackoverflow.com/a/25859000/5955544
// SFINAE test

template <typename T>
using remove_ref_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_refptr_t = typename std::remove_pointer<remove_ref_t<T>>::type;

template <typename T>
using is_function_t = typename std::is_function<remove_refptr_t<T>>::type;

////// SFINAE

template <typename T>
class is_invocable
{

    typedef char one;
    typedef struct
    {
        char a[2];
    } two;

    template <typename C>
    static one test(decltype(&C::operator())); // decltype(C::helloworld)*
    template <typename C>
    static two test(...);

    public:
    enum
    {
        value = sizeof(test<T>(0)) == sizeof(char)
    };
}; //    std::cout << is_invocable<PotentiallyInvocable>::value << std::endl;

template <typename T>
bool check_invocable(const T &a)
{
    return is_invocable<T>::value;
}

////// wrapper for lookup or creation of logger
std::shared_ptr<spdlog::logger>
logger_checkin(std::string name); // not really working

////// https://stackoverflow.com/a/12399290/5955544
// TODO templatize on parameter (not in use now)
template <typename Derived>
std::vector<Index> sort_indices_decending(const Derived &col)
{ // std::vector<T>

    // initialize original index locations
    std::vector<Index> idx(col.rows());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indices based on comparing values in col
    sort(idx.begin(), idx.end(),
         [&col](Index i1, Index i2) { return col[i1] > col[i2]; });

    return idx;
}

} // namespace utilities
} // namespace mimkl

#endif /* INCLUDE_MIMKL_UTILITIES_HPP_ */
