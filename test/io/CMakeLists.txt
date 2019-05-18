add_executable(io.exe main.cpp)
target_link_libraries(io.exe LINK_PUBLIC mimkl)
add_test(NAME io COMMAND io.exe)
