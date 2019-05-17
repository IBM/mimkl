add_executable(linear_algebra.exe main.cpp)
target_link_libraries(linear_algebra.exe LINK_PUBLIC mimkl)
add_test(NAME linear_algebra COMMAND linear_algebra.exe)
