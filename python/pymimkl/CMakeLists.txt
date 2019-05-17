pybind11_add_module(_pymimkl wrapper.cpp)
target_link_libraries(_pymimkl LINK_PUBLIC mimkl)
file(COPY __init__.py DESTINATION .)
file(COPY easy_mkl.py DESTINATION .)
file(COPY umkl_knn.py DESTINATION .)
file(COPY average_mkl.py DESTINATION .)
file(COPY utils.py DESTINATION .)

# ./tests/* files are here for ` python -m unittest` to be discovered
# but destination is test for `make test`
# also, these are not installed in the python module (not copied to build dir)
file(COPY tests/test.py DESTINATION ../../../test/pymimkl)