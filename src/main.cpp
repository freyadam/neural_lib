
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

    nl::CsvReader r("reader", "test/csv/valid.csv");

    nl::Block* o1 = r.outputs()["reader_out0"];
    nl::Block* o2 = r.outputs()["reader_out1"];

    for (int i = 0; i < 10; i++) {
        r.forward();
        std::cout << o1->data(0,0,0) << "/" << o2->data(0,0,0) << std::endl;
    }

    return 0;
}
