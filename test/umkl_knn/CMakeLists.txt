add_executable(umkl_knn.exe main.cpp)
target_link_libraries(umkl_knn.exe LINK_PUBLIC mimkl)
add_test(NAME umkl_knn COMMAND umkl_knn.exe)
