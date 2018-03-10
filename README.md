# neural_lib

neural_lib is a C++ library for feed-forward networks.

## External dependencies

Several external projects or libraries are used to provide functionality that is not the focus of neural_lib. Namely they are `CImg` for image loading and processing, `Eigen` for matrix computation and `boost::serialization` for serialization of neural networks. Both `CImg` and `Eigen` do not need to be compiled as they are header-only projects. 

All dependencies are provided in the `extern` folder for ease of use.

## Usage

The precise usage depends on the usage. By default, project is compiled as a dynamical library in the `bin` directory using the `make all` command. 

## Tests

This project is using the Google Test framework.

Build using the commands:

```bash
cd extern/googletest
mkdir lib
cd lib
cmake ..
make
```

Alternativelly if you have the framework already installed on your computer, you can just use it by redefining the `GTEST_DIR` variable in `test/Makefile`.

## Documentation

Source code is annotated with the Doxygen-style comments. Doxygen documentation can be created with `make doc`. Result can be then found in the `doc` directory.