# neural_lib

neural_lib is a C++ library for feed-forward neural networks. The structure of neural network is created from two basic components, block and operation (op for short). Blocks serve as data storage both for signal being passed forward and for gradient being propagated backwards. Operations do not hold any data on their own but instead specify local computation. During forward pass they modify their outputs based on their inputs and during backward pass vice versa. While blocks are quite similar to each other in almost all aspects apart from dimension, there are several operations implemented. Namely they are `Neuron`, representing a single artificial neuron, `Dense` (fully connected) layer, `Conv`, convolutional layer, and `MaxPool`, max pooling layer.

In addition to ops that compute their output based on the content of input blocks, there are also implemented two input operations. `CsvReader` takes a csv file as an input and during each forward pass, saves a content of single line to its output block. This way it sequentially processes the whole list. When it reaches the end, it starts again from the beginning. `ImgReader` is instead created with a file with image addresse on each line. During each forward pass, it loads the current image to the output block. Similarly to `CsvReader`, it also restarts when end of file is reached. As blocks have a fixed dimension, it is important that all images have the same dimensions.

As it must be the case with all feed-forward neural networks, operations must form directed acyclic graph. To simplify usage of the library, a Net class is implemented that is used to store operations and blocks and create directed acyclic graph on its own. This allows it to call operations in correct order. 

While it is possible to modify weights and thresholds manually, Solver class is implemented that allows for gradient descent supervised learning. 

To illustrate the functionality of this library, several examples are implemented. `example1.cpp` contains basic demonstration of library functionality by training a neuron for linear separation. `example2.cpp` shows how to manually modify weights, use reader ops and create more complex networks. Its corresponding network learns XOR function.

Networks are also capable of serialization using the `boost::serialization` library. It's usage is for example well described in [`A Very Simple Case`](https://www.boost.org/doc/libs/1_66_0/libs/serialization/doc/tutorial.html) code snippet of `boost::serialization` documentation.

This readme is obviously too short for any meaningful introduction to the topic of neural networks. For good introduction to neural networks, check for example the [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) course notes from Stanford that cover everything in this library.

## External dependencies

Several external projects or libraries are used to provide functionality that is not the focus of neural_lib. Namely they are `CImg` for image loading and processing, `Eigen` for matrix computation and `boost::serialization` for serialization of neural networks. Both `CImg` and `Eigen` do not need to be compiled as they are header-only projects. 

All dependency headers from these projects are provided in the `extern` folder for ease of use.

As a sidenote, to make Eigen folder header-only its structure was changed a bit. Naturally, some files were left out but some files were also moved around. For that reason the addresses to files in `extern/Eigen` folder and the addresses in original project may not correspond.

## Usage

The precise usage depends on the use case. By default, project is compiled as a dynamical library in the `bin` directory using the `make all` command. Currently library is called `libneural`. Running `make run` in addition to building the library compiles `main.cpp` as independent program and runs it with neural library linked. Similarly `run_ex1` and `run_ex2` build all prerequisites and run either `example1` or `example2`.

## Tests

This project is using the Google Test framework, which can be built using the commands:

```bash
cd extern/googletest
mkdir lib
cd lib
cmake ..
make
```

Alternativelly if you have the framework already installed on your computer, you can use it by redefining the `GTEST_DIR` variable in `test/Makefile`.

To run all tests, just type 'make test' in main project directory. Currently tests are stored in `hpp` header files in the directory `test`. To add new tests just add code to one of these files or create a new one. If new test header file is created, include it in `test/test.cpp`.

## Documentation

Source code is annotated with Doxygen-style comments. Doxygen documentation can be created with `make doc`. Result can be then found in the `doc` directory.
