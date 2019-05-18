add_executable(average_mkl.exe main.cpp)
target_link_libraries(average_mkl.exe LINK_PUBLIC mimkl)
add_test(NAME average_mkl COMMAND average_mkl.exe)
