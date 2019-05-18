add_executable(kernels_handler.exe main.cpp)
target_link_libraries(kernels_handler.exe LINK_PUBLIC mimkl)
add_test(NAME kernels_handler COMMAND kernels_handler.exe)
