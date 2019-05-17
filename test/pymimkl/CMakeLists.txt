# python/pymimkl/CMakeLists.txt copies files here
add_test(NAME pymimkl 
         COMMAND ${PYTHON_EXECUTABLE} -m unittest discover -s .)
set_tests_properties(
    pymimkl PROPERTIES
    ENVIRONMENT "PYTHONPATH=${mimkl_BINARY_DIR}/python;${PYTHONPATH}"
)  # built python module is found when on PYTHONPATH