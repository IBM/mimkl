add_executable(easy_mkl.exe main.cpp)
target_link_libraries(easy_mkl.exe LINK_PUBLIC mimkl)
add_test(NAME easy_mkl COMMAND easy_mkl.exe)
