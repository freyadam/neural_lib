
#include <iostream>
#include <random>
#include <string> 

#include <Eigen/unsupported/CXX11/Tensor>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"
#include "neuron.hpp"
#include "reader.hpp"
#include "error.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::ImgReader r("reader", "test/img/valid.csv");
    r.forward();

    return 0;
}
