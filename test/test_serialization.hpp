
#ifndef NEURAL_LIB_SERIALIZATION_TEST_H
#define NEURAL_LIB_SERIALIZATION_TEST_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <string> 
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

TEST(SerializationTest, Block) {

    // create block
    nl::Block b("b", 1, 1, 2);
    b.trainable = false;
    b.data(0,0,0) = 3.14;
    b.data(0,0,1) = 12e-3;
    b.grad(0,0,0) = -1;
    b.grad(0,0,1) = 3e12;

    // save block
    std::string filename = "test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << b;
    }
    // load block
    nl::Block b2("b2", 1, 1, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> b2; 
    }

    auto dim1 = b.dimensions();
    auto dim2 = b2.dimensions();

    EXPECT_EQ(dim1[0], dim2[0]);
    EXPECT_EQ(dim1[1], dim2[1]);
    EXPECT_EQ(dim1[2], dim2[2]);

    EXPECT_EQ(b.name, b2.name);
    EXPECT_EQ(b.trainable, b2.trainable);
    
    EXPECT_FLOAT_EQ(b.data(0,0,0), b2.data(0,0,0));
    EXPECT_FLOAT_EQ(b.data(0,0,1), b2.data(0,0,1));
    EXPECT_FLOAT_EQ(b.grad(0,0,0), b2.grad(0,0,0));
    EXPECT_FLOAT_EQ(b.grad(0,0,1), b2.grad(0,0,1));

}

TEST(SerializationTest, Dense) {
    
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
    
    // inputs
    EXPECT_EQ(d2.inputs().size(), 3);
    EXPECT_NE(d2.inputs()["b"], nullptr);
    EXPECT_NE(d2.inputs()["d1_w"], nullptr);
    EXPECT_NE(d2.inputs()["d1_thr"], nullptr);

    // outputs
    EXPECT_EQ(d2.outputs().size(), 1);
    EXPECT_NE(d2.outputs()["d1_out"], nullptr);


    d2.inputs()["b"]->data(0,0,0) = 1;
    d2.forward();
    EXPECT_EQ(d2.outputs()["d1_out"]->data(0,0,0), 5);
    
    d2.inputs()["b"]->data(0,0,0) = -1;
    d2.forward();
    EXPECT_EQ(d2.outputs()["d1_out"]->data(0,0,0), 0);

}

TEST(SerializationTest, Net) {

    nl::Block b("b", 1, 1, 1);
    nl::Neuron n1("n1", "linear", &b);
    nl::Neuron n2("n2", "relu", &n1);

    // set weights
    nl::Block* thr1 = n1.inputs()["n1_thr"];
    nl::Block* w1 = n1.inputs()["n1_b_w"];
    nl::Block* thr2 = n2.inputs()["n2_thr"];
    nl::Block* w2 = n2.inputs()["n2_n1_out_w"];
    thr1->data(0,0,0) = 1;
    w1->data(0,0,0) = 3;
    thr2->data(0,0,0) = 0;
    w2->data(0,0,0) = 1;

    nl::Net net("net");
    net.add(&n2);
    net.add(&n1);

    // save net
    std::string filename = "net_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << net;
    }
    // load net
    nl::Net net2("net2");
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> net2; 
    }

    // net name
    EXPECT_STREQ(net2.name.c_str(), "net");

    // inputs
    EXPECT_EQ(net2.inputs().size(), 5);
    EXPECT_NE(net2.inputs()["n1_thr"], nullptr);
    EXPECT_NE(net2.inputs()["n1_b_w"], nullptr);
    EXPECT_NE(net2.inputs()["n2_n1_out_w"], nullptr);
    EXPECT_NE(net2.inputs()["n2_thr"], nullptr);
    EXPECT_NE(net2.inputs()["b"], nullptr);

    // outputs
    EXPECT_EQ(net2.outputs().size(), 1);
    EXPECT_NE(net2.outputs()["n2_out"], nullptr);

    nl::Block* in = net2.inputs()["b"];

    in->data(0,0,0) = 2;
    net2.forward();
    EXPECT_FLOAT_EQ(net2.outputs()["n2_out"]->data(0,0,0), 7);

    in->data(0,0,0) = -100;
    net2.forward();
    EXPECT_FLOAT_EQ(net2.outputs()["n2_out"]->data(0,0,0), 0);

}

#endif // NEURAL_LIB_SERIALIZATION_TEST_H
