
#include <iostream>
#include <random>
#include <string> 

#include "neural.hpp"

using namespace std;

/*
Example: Perceptron (linear separability) learning

Single neuron with tanh transfer function is here used to approximate
a hyperplane. This example is designed to show certain basic 
features of the library. In the beginning, input blocks and used 
neuron is created. That neuron is then inserted into a network.
This network is then trained using an instance of a Solver class.

This example is quite basic so no file or image I/O is performed
and input is generated and inserted from outside of the library.
For that reason solver.train() is called which runs only a single
cycle of forward pass, backward pass and weight modification.
solver.train(5) on the other hand would run 5 cycles.
*/

int main(int argc, char *argv[])
{
    
    // create blocks of dimensions [1,1,1]
    // in which input is stored
    nl::Block b1("x", 1,1,1); 
    nl::Block b2("y", 1,1,1);
    // create block with desired value
    nl::Block d("z", 1,1,1);

    // create perceptron with tanh transfer function 
    nl::Neuron n("n", "tanh", &b1, &b2);

    // block with output of neuron
    nl::Block &o = *n.outputs()["n_out"];

    // neuron weights
    // nl::Block &w1 = *n.inputs()["n_b1_w"];
    // nl::Block &w2 = *n.inputs()["n_b2_w"];
    // nl::Block &thr = *n.inputs()["n_thr"];

    // create network and put perceptron inside
    nl::Net net("net");
    net.add(&n);

    // create solver that is supposed to train the network
    // 'o' specifies the net result
    // 'd' specified the desired output
    nl::Solver solver(&net, &o, &d);
    // solver.setMethod("nesterov"); // use training with momentum

    // define hyperplane parameters that should be learned
    float a=3, b=-2, c=4;
    float x, y, z;

    // train net for given number of iterations
    uint16_t iterations = 100;
    for (uint16_t i = 0; i < iterations; ++i) {

        // generate and 'manually' load input into blocks
        b1.data(0,0,0) = x = 10 * nl::Generator::get();
        b2.data(0,0,0) = y = 10 * nl::Generator::get();
        d.data(0,0,0) = z = (a*x + b*y - c > 0 ? 1 : -1);

        // run single train cycle that runs forward, backward on net
        // and then updates the weights
        std::cout << solver.train() << std::endl;        
    }

    return 0;
}
