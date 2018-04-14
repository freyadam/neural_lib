
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
    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 2);
    b->trainable = false;
    b->data(0,0,0) = 3.14;
    b->data(0,0,1) = 12e-3;
    b->grad(0,0,0) = -1;
    b->grad(0,0,1) = 3e12;

    // save block
    std::string filename = "test/block_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << b;
    }
    // load block
    nl::block_ptr b2 = std::make_shared<nl::Block>("b2", 1, 1, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> b2; 
    }

    auto dim1 = b->dimensions();
    auto dim2 = b2->dimensions();

    EXPECT_EQ(dim1[0], dim2[0]);
    EXPECT_EQ(dim1[1], dim2[1]);
    EXPECT_EQ(dim1[2], dim2[2]);

    EXPECT_EQ(b->name, b2->name);
    EXPECT_EQ(b->trainable, b2->trainable);
    
    EXPECT_FLOAT_EQ(b->data(0,0,0), b2->data(0,0,0));
    EXPECT_FLOAT_EQ(b->data(0,0,1), b2->data(0,0,1));
    EXPECT_FLOAT_EQ(b->grad(0,0,0), b2->grad(0,0,0));
    EXPECT_FLOAT_EQ(b->grad(0,0,1), b2->grad(0,0,1));

}

TEST(SerializationTest, Dense) {
    
    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    nl::Dense d1("d1", "relu", b,
                 2, 2, 2);

    d1.inputs()["d1_w"]->data(0,0,0) = 3;
    d1.inputs()["d1_thr"]->data(0,0,0) = 2;     

    // save layer
    std::string filename = "test/dense_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << d1;
    }
    // load layer
    nl::Dense d2("d2", "linear", b,
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

TEST(SerializationTest, Conv) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    nl::Conv c1("c1", "relu", b,
                 2, 3, 1);

    c1.inputs()["c1_w1"]->data(0,1,1) = 3;
    c1.inputs()["c1_thr1"]->data(0,0,0) = 2;     

    // save layer
    std::string filename = "test/conv_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << c1;
    }
    // load layer
    nl::Conv c2("c2", "linear", b,
                 1, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> c2; 
    }

    // net name
    EXPECT_STREQ(c2.name.c_str(), "c1");

    // inputs
    EXPECT_EQ(c2.inputs().size(), 5);
    EXPECT_NE(c2.inputs()["c1_thr0"], nullptr);
    EXPECT_NE(c2.inputs()["c1_w0"], nullptr);
    EXPECT_NE(c2.inputs()["c1_thr1"], nullptr);
    EXPECT_NE(c2.inputs()["c1_w1"], nullptr);
    EXPECT_NE(c2.inputs()["b"], nullptr);

    // outputs
    EXPECT_EQ(c2.outputs().size(), 1);
    EXPECT_NE(c2.outputs()["c1_out"], nullptr);

    // defined weights
    EXPECT_EQ(c2.inputs()["c1_w1"]->data(0,1,1), 3);
    EXPECT_EQ(c2.inputs()["c1_thr1"]->data(0,0,0), 2);

    c2.inputs()["b"]->data(0,0,0) = 1;
    c2.forward();
    EXPECT_FLOAT_EQ(c2.outputs()["c1_out"]->data(1,0,0), 5);

    c2.inputs()["b"]->data(0,0,0) = -1;
    c2.forward();
    EXPECT_FLOAT_EQ(c2.outputs()["c1_out"]->data(1,0,0), 0);
}

TEST(SerializationTest, MaxPool) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 2, 2);
    nl::MaxPool m1("m1", b, 2);

    // save layer
    std::string filename = "test/maxpool_serialization_test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << m1;
    }
    // load layer
    nl::MaxPool m2("m2", b, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> m2; 
    }

    // inputs
    EXPECT_EQ(m2.inputs().size(), 1);
    EXPECT_NE(m2.inputs()["b"], nullptr);

    // outputs
    EXPECT_EQ(m2.outputs().size(), 1);
    EXPECT_NE(m2.outputs()["m1_out"], nullptr);
        
    nl::block_ptr out = m2.outputs()["m1_out"];
    auto out_dims = out->dimensions();
    EXPECT_EQ(out_dims[0], 1);
    EXPECT_EQ(out_dims[1], 1);
    EXPECT_EQ(out_dims[2], 1);

    m2.inputs()["b"]->data(0,0,0) = 1;
    m2.inputs()["b"]->data(0,0,1) = -1;
    m2.inputs()["b"]->data(0,1,0) = 3;
    m2.inputs()["b"]->data(0,1,1) = -1e3;
    m2.forward();
    EXPECT_EQ(m2.outputs()["m1_out"]->data(0,0,0), 3);
}

TEST(SerializationTest, Net) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    nl::Neuron n1("n1", "linear", b);
    nl::Neuron n2("n2", "relu", &n1);

    // set weights
    nl::block_ptr thr1 = n1.inputs()["n1_thr"];
    nl::block_ptr w1 = n1.inputs()["n1_b_w"];
    nl::block_ptr thr2 = n2.inputs()["n2_thr"];
    nl::block_ptr w2 = n2.inputs()["n2_n1_out_w"];
    thr1->data(0,0,0) = 1;
    w1->data(0,0,0) = 3;
    thr2->data(0,0,0) = 0;
    w2->data(0,0,0) = 1;

    nl::Net net("net");
    net.add(&n2);
    net.add(&n1);

    // save net
    std::string filename = "test/net_serialization_test.txt";
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

    nl::block_ptr in = net2.inputs()["b"];

    in->data(0,0,0) = 2;
    net2.forward();
    EXPECT_FLOAT_EQ(net2.outputs()["n2_out"]->data(0,0,0), 7);

    in->data(0,0,0) = -100;
    net2.forward();
    EXPECT_FLOAT_EQ(net2.outputs()["n2_out"]->data(0,0,0), 0);

}

TEST(SerializationTest, CsvReader) {


    nl::CsvReader r("reader", "test/csv/valid.csv");
    nl::Net n("net");
    n.add(&r);
    
    // save net
    std::string filename = "test/csvreader_serialization_test.txt";
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
    
    EXPECT_EQ(n2.name, "net");
    EXPECT_NE(n2.outputs()["reader_out"], nullptr);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,0,1), 2.3);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,0,1), -6);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,0,1), -1.23e-1);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,0,1), 2.3);

}


TEST(SerializationTest, ImgReader) {

    nl::ImgReader r("reader", "test/img/valid.csv");
    nl::Net n("net");
    n.add(&r);
    
    // save net
    std::string filename = "test/imgreader_serialization_test.txt";
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
    
    EXPECT_EQ(n2.name, "net");
    EXPECT_NE(n2.outputs()["reader_out"], nullptr);
        
    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,12,12), 255);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,12,12), 183);

    n2.forward();
    EXPECT_FLOAT_EQ(n2.outputs()["reader_out"]->data(0,12,12), 255);
}


#endif // NEURAL_LIB_SERIALIZATION_TEST_H
