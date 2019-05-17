add_executable(gaussian_induction.exe main.cpp)
target_link_libraries(gaussian_induction.exe)
add_test(NAME gaussian_induction COMMAND gaussian_induction.exe)
