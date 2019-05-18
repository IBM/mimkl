add_executable(kernel_lambdas.exe main.cpp)
target_link_libraries(kernel_lambdas.exe LINK_PUBLIC mimkl)
add_test(NAME kernel_lambdas COMMAND kernel_lambdas.exe)
