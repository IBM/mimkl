add_executable(data_frame.exe main.cpp)
target_link_libraries(data_frame.exe LINK_PUBLIC mimkl)
add_test(NAME data_frame COMMAND data_frame.exe)
