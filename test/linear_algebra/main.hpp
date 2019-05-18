#ifndef TEST_LINEAR_ALGEBRA_MAIN_HPP_
#define TEST_LINEAR_ALGEBRA_MAIN_HPP_

#include <dlib/optimization.h>
#include <spdlog/spdlog.h>

template <typename T, long NR, long NC, typename MM, typename L>
void change_dlib_mat(dlib::matrix<T, NR, NC, MM, L> &dlib_mat)
{
    dlib_mat(0, 0) = dlib_mat(0, 0) + 1.1;
    spdlog::get("console")->info("one value changed");
}

#endif /* TEST_LINEAR_ALGEBRA_MAIN_HPP_ */
