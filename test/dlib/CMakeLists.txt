add_executable(dlib.exe main.cpp)
target_link_libraries(dlib.exe LINK_PUBLIC mimkl)
add_test(NAME dlib COMMAND dlib.exe)
