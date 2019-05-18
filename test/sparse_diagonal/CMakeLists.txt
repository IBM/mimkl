add_executable(sparse_diagonal.exe main.cpp)
target_link_libraries(sparse_diagonal.exe)
add_test(NAME sparse_diagonal COMMAND sparse_diagonal.exe)
