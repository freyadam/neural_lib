
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
    nl::Dense d("d", "relu", &r, 3, 5, 7);
    nl::Net n("net");
    n.add(&r);
    n.add(&d);

    return 0;
}
