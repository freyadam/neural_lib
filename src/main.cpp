
#include <iostream>
#include <random>

#include <Eigen/unsupported/CXX11/Tensor>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"
#include "neuron.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Block b1("block", 1, 1, 1);
    nl::Block b2("block", 1, 1, 1);
    nl::Neuron n("n", "relu", &b1, &b2);

    // get neuron output
    nl::Block * out = n.outputs()["n_out"];

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 0;    
    n.forward();

    float thr = out->data(0,0,0);
    
    b1.data(0,0,0) = 1;
    n.forward();

    float w1 = out->data(0,0,0) +  thr;

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 1;    
    n.forward();

    float w2 = out->data(0,0,0) +  thr;

    std::cout << "thr: " << thr << std::endl;
    std::cout << "w1: " << w1 << std::endl;
    std::cout << "w2: " << w2 << std::endl;

    return 0;
}
