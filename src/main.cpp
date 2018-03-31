
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <string> 
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/base_object.hpp>

#include "neural.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    
    // nl::Block b("b", 1, 1, 1);
    // nl::Neuron n("n", "relu", &b);
    
    // // save block
    // std::string filename = "test.txt";
    // {
    //     std::ofstream ofs(filename);
    //     boost::archive::text_oarchive oa(ofs);
    //     oa << n;
    // }
    // // load block
    // nl::Neuron n2("n2", "linear", &b);
    // {
    //     std::ifstream ifs(filename);
    //     boost::archive::text_iarchive ia(ifs);
    //     ia >> n2; 
    // }

    // std::cout << n2.name << std::endl;
    // std::cout << "inputs:" << std::endl;
    // for (auto pair : n2.inputs()) {
    //     std::cout << pair.second->name << std::endl;
    // }
    // std::cout << "outputs:" << std::endl;
    // for (auto pair : n2.outputs()) {
    //     std::cout << pair.second->name << std::endl;
    
    // }
    // n2.inputs()["n_thr"]->data(0,0,0) = 0;
    // n2.inputs()["n_b_w"]->data(0,0,0) = 1;
    // nl::Block* in = n2.inputs()["b"];

    // in->data(0,0,0) = 0.37;
    // n2.forward();
    // std::cout << "forward: " << n2.outputs()["n_out"]->data(0,0,0) << std::endl;

    // in->data(0,0,0) = -100;
    // n2.forward();
    // std::cout << "forward: " << n2.outputs()["n_out"]->data(0,0,0) << std::endl;

    return 0;
}
