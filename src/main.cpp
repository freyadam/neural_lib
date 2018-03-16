
#include <iostream>
#include <random>
#include <string> 

#include <Eigen/unsupported/CXX11/Tensor>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"
#include "neuron.hpp"
#include "reader.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n("n", "linear", &b1, &b2);

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 0;
    n.forward();

    float threshold = n.outputs()["n_out"]->data(0,0,0);

    b1.data(0,0,0) = 1;
    b2.data(0,0,0) = 0;
    n.forward();

    float w1 = n.outputs()["n_out"]->data(0,0,0) - threshold;

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 1;
    n.forward();

    float w2 = n.outputs()["n_out"]->data(0,0,0) - threshold;

    std::cout << "threshold: " << threshold << std::endl;
    std::cout << "w1: " << w1 << std::endl;
    std::cout << "w2: " << w2 << std::endl;

    return 0;
}
