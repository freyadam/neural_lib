
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <string> 
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/export.hpp>

#include "neural.hpp"

using namespace std;

int main(int argc, char *argv[])
{

    nl::ImgReader r("reader", "test/img/valid.csv");
    nl::Net n("net");
    n.add(&r);
    
    // save net
    std::string filename = "imgreader_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << n;
    }
    // load net
    nl::Net n2("net2");
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> n2;               
    }
    
    std::cout << n2.name << std::endl;
    std::cout << n2.outputs()["reader_out"] << std::endl;    
        
    n2.forward();
    std::cout << n2.outputs()["reader_out"]->data(0,12,12) << std::endl;

    n2.forward();
    std::cout << n2.outputs()["reader_out"]->data(0,12,12) << std::endl;

    n2.forward();
    std::cout << n2.outputs()["reader_out"]->data(0,12,12) << std::endl;

    return 0;
}
