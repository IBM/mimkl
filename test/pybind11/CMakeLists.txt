pybind11_add_module(example example.cpp example_wrapper.cpp)
target_link_libraries(example LINK_PUBLIC mimkl)
file(COPY example.py DESTINATION .)
add_test(NAME pybind11 COMMAND ${PYTHON_EXECUTABLE} example.py)
