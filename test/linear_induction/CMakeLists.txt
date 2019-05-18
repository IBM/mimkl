add_executable(linear_induction.exe main.cpp)
target_link_libraries(linear_induction.exe)
add_test(NAME linear_induction COMMAND linear_induction.exe)
