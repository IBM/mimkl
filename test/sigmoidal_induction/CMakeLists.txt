add_executable(sigmoidal_induction.exe main.cpp)
target_link_libraries(sigmoidal_induction.exe)
add_test(NAME sigmoidal_induction COMMAND sigmoidal_induction.exe)
