add_executable(polynomial_induction.exe main.cpp)
target_link_libraries(polynomial_induction.exe)
add_test(NAME polynomial_induction COMMAND polynomial_induction.exe)
