
#include <iostream>
#include <random>
#include <string> 

// #include <Eigen/unsupported/CXX11/Tensor>

#include "neural.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Block b("b", 1, 2, 2);
    nl::Softmax s("s", &b);
    
    nl::Block* b2 = s.outputs()["s_out"];

    b.data(0,0,0) = 5;
    b.data(0,0,1) = 0;
    b.data(0,1,0) = 2;
    b.data(0,1,1) = 0;

    s.forward();

    std::cout << b2->data(0,0,0) << std::endl;
    std::cout << b2->data(0,0,1) << std::endl;
    std::cout << b2->data(0,1,0) << std::endl;
    std::cout << b2->data(0,1,1) << std::endl;

    // nl::Block b1("b1", 1,1,1);
    // nl::Block b2("b2", 1,1,1);
    // nl::Block d("d", 1,1,1);

    // nl::Neuron n("n", "tanh", &b1, &b2);
    // nl::Block &o = *n.outputs()["n_out"];
    // nl::Block &w1 = *n.inputs()["n_b1_w"];
    // nl::Block &w2 = *n.inputs()["n_b2_w"];
    // nl::Block &thr = *n.inputs()["n_thr"];

    // float a=5, b=-2, c=3;
    // float x, y, z;

    // for (uint16_t i = 0; i < 10000; ++i) {
    //     n.zero_grad();

    //     b1.data(0,0,0) = x = 10 * nl::Generator::get();
    //     b2.data(0,0,0) = y = 10 * nl::Generator::get();
    //     d.data(0,0,0) = z = (a*x + b*y - c > 0 ? 1 : -1);

    //     // forward pass
    //     n.forward();
    //     // compute error
    //     o.grad(0,0,0) = nl::Error::L2_grad(&o, &d);
    //     // print error
    //     std::cout << "err: " << nl::Error::L2(&o, &d) << " (" 
    //               << w1.data(0,0,0) << "/" 
    //               << w2.data(0,0,0) << "/" 
    //               << thr.data(0,0,0) << ")" 
    //               <<std::endl;
    //     // backward pass
    //     n.backward();        
    //     // update
    //     w1.data -= 0.05 * w1.grad;
    //     w2.data -= 0.05 * w2.grad;
    //     thr.data -= 0.05 * thr.grad;
    // }

    return 0;
}
