
#include <iostream>

#include <Eigen/unsupported/CXX11/Tensor>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"
#include "perceptron.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Perceptron p("test", 1, 2, 3, 4, 5);

    return 0;
}
