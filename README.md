[![Build Status](https://travis-ci.org/IBM/mimkl.svg?branch=master)](https://travis-ci.org/IBM/mimkl)

# mimkl

A C++ library for Matrix-Induced Multiple Kernel Learning (MIMKL) with python bindings. The library implements unsupervised and supervised multiple kernel learning algorithms and the most common kernel functions as well as matrix-inducing versions thereof. The python wrapper exposes classes with interfaces that allow a seamless integration with scikit-learn.

The implementation has been used to generate all the results for the paper ["Pathway-Induced Multiple Kernel Learning"](https://www.nature.com/articles/s41540-019-0086-3).

## Requirements

- C++14 capable C++ compiler
- cmake (>3.0.2)
- Python

## pymimkl

`pymimkl` is the python package for matrix induced multiple kernel learning. It uses bindings from compiled mimkl (C++) (see [Building](##Building)) and provides scikit-learn like classifiers.

### Installation

Install directly from git.
This builds the mimkl project and the pymimkl bindings and might take some time.

```sh
pip install git+ssh://git@github.com/IBM/mimkl.git
```

Optimize the build by passing additional C++ flags.
For example to enable parallelism (compiler must support OpenMP):

```sh
MIMKL_CXX_FLAGS="-fopenmp" pip install git+ssh://git@github.com/IBM/mimkl.git
```

Enable architecture optimizations (i386):

```sh
MIMKL_CXX_FLAGS="-march=native" pip install git+ssh://git@github.com/IBM/mimkl.git
```

Enable both:

```sh
MIMKL_CXX_FLAGS="-march=native -fopenmp" pip install git+ssh://git@github.com/IBM/mimkl.git
```

### Docker

Pull the image from DockerHub:

```sh
docker pull tsenit/mimkl
```

Or directly build it from source:

```
docker-compose -f docker/docker-compose.yml build
```

Run the container: 

```
docker run -it tsenit/mimkl /bin/bash
```

### Development

Clone the repository:

```sh
git clone https://github.com/IBM/mimkl
```

Intialize and update the submodules

```sh
cd mimkl
git submodule init
git submodule update
```

Install `pymimkl` with pip.

```sh
pip install .
# to persist the created build_ folder for faster rebuilding and C++ testing:
# pip install -e .
```

Same as before to optimize the build.

```sh
MIMKL_CXX_FLAGS="-march=native -fopenmp" pip install .
```

run tests with

```sh
python setup.py test
```

## Building

Clone the repository:

```sh
git clone https://github.com/IBM/mimkl
```

Intialize and update the submodules

```sh
cd mimkl
git submodule init
git submodule update
```

Create a build folder (in source build):

```sh
mkdir build
cd build
```

Generate the building files:

```sh
cmake ..
# unfortuantely, cmake is not aware of virtual environments. The fix is
# cmake -DPYTHON_EXECUTABLE=$(command -v python) ..
# https://github.com/pybind/pybind11/issues/99
```

Compile the code:

```sh
make
```

Test it:

```sh
make test
# make test CTEST_OUTPUT_ON_FAILURE=TRUE
```

Take note that the python tests have additional requirements (e.g. scipy).

# References

## Publications

[EasyMKL: a scalable multiple kernel learning algorithm](https://doi.org/10.1016/j.neucom.2014.11.078)

[Unsupervised multiple kernel learning for heterogeneous data integration](https://doi.org/10.1093/bioinformatics/btx682)  ("UMKLKNN" im mimkl)

## Code

eigen [http://eigen.tuxfamily.org/index.php?title=Main_Page](http://eigen.tuxfamily.org/index.php?title=Main_Page)

dlib [http://dlib.net](http://dlib.net)

pybind11 [https://github.com/pybind/pybind11](https://github.com/pybind/pybind11)

## Uses of mimkl

PIMKL: Pathway-Induced Multiple Kernel Learning [https://rdcu.be/bBN6U](https://rdcu.be/bBN6U)  
for which mimkl was developed.

<p align="center">
  <img src="doc/figures/pimkl.png" alt="PIMKL algorithm" width=700>
</p>
