
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <string> 
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "neural.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::Block b("b", 1, 1, 1);
    nl::Dense d1("d1", "relu", &b,
                 2, 2, 2);

    d1.inputs()["d1_w"]->data(0,0,0) = 3;
    d1.inputs()["d1_thr"]->data(0,0,0) = 2;     

    // save layer
    std::string filename = "test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << d1;
    }
    // load layer
    nl::Dense d2("d2", "linear", &b,
                 1, 1, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> d2; 
    }
        
    d2.inputs()["b"]->data(0,0,0) = 1;
    d2.forward();
    std::cout << d2.outputs()["d1_out"]->data(0,0,0) << std::endl;

    d2.inputs()["b"]->data(0,0,0) = -1;
    d2.forward();
    std::cout << d2.outputs()["d1_out"]->data(0,0,0) << std::endl;
        
    return 0;
}
