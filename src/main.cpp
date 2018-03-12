
#include <iostream>

#include <Eigen/unsupported/CXX11/Tensor>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"
#include "neuron.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Block b("block", 1, 1, 1);
    nl::Neuron n("test", &b);

    return 0;
}
