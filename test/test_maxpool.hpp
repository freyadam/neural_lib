#ifndef NEURAL_LIB_MAXPOOL_TEST_H
#define NEURAL_LIB_MAXPOOL_TEST_H

#include "exceptions.hpp"
#include "maxpool.hpp"

TEST(MaxPoolTest, Constructor) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    nl::Neuron n("n", "relu", b);
    nl::MaxPool p("pool", n, 2, 1);

    // check that max pool has correct input block
    EXPECT_EQ(p.inputs().begin()->second->name, "n_out");

    // padding too large
    EXPECT_THROW(nl::MaxPool("p", b, 2, 2), nl::InputException);

    // window too large
    EXPECT_THROW(nl::MaxPool("p", b, 2), nl::InputException);
    EXPECT_THROW(nl::MaxPool("p", b, 6, 2), nl::InputException);

    // output of correct size
    EXPECT_EQ(p.inputs().size(), 1);
    EXPECT_EQ(p.outputs().size(), 1);    

}

TEST(MaxPoolTest, Forward) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    b->data(0,0,0) = 3.14;

    nl::MaxPool p("p", b, 2, 1);
    nl::block_ptr p_out = p.outputs()["p_out"];
    p.forward();
    // all values need to equal 3.14
    EXPECT_FLOAT_EQ(p_out->data(0,0,0), 3.14);
    EXPECT_FLOAT_EQ(p_out->data(0,0,1), 3.14);
    EXPECT_FLOAT_EQ(p_out->data(0,1,0), 3.14);
    EXPECT_FLOAT_EQ(p_out->data(0,1,1), 3.14);
        
    nl::block_ptr b2 = std::make_shared<nl::Block>("b2", 2, 2, 2);
    // first depth slice
    b2->data(0,0,0) = 3.14;    
    b2->data(0,0,1) = -3.14;    
    b2->data(0,1,0) = 0;    
    b2->data(0,1,1) = 1000;    
    // second depth slice
    b2->data(1,0,0) = -3.14e2;    
    b2->data(1,0,1) = -3.14;    
    b2->data(1,1,0) = -50;    
    b2->data(1,1,1) = -256;    
    nl::MaxPool p2("p2", b2, 2);
    nl::block_ptr p2_out = p2.outputs()["p2_out"];
    p2.forward();

    EXPECT_FLOAT_EQ(p2_out->data(0,0,0), 1000);
    EXPECT_FLOAT_EQ(p2_out->data(1,0,0), -3.14);        

}

TEST(MaxPoolTest, Backward) {
       
    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 1, 1);
    b->data(0,0,0) = 3.14;

    nl::MaxPool p("p", b, 2, 1);
    nl::block_ptr p_out = p.outputs()["p_out"];
    p_out->grad(0,0,0) = 1;
    p_out->grad(0,0,1) = 1;
    p_out->grad(0,1,0) = 1;
    p_out->grad(0,1,1) = 1;

    p.backward();

    EXPECT_FLOAT_EQ(b->grad(0,0,0), 4);

    std::cout << "-----" << std::endl;

    nl::block_ptr b2 = std::make_shared<nl::Block>("b2", 1, 2, 2);
    b2->data(0,0,0) = 3.14;
    b2->data(0,0,1) = 0;
    b2->data(0,1,0) = -5;
    b2->data(0,1,1) = -1e17;
    nl::MaxPool p2("p2", b2, 2);
    nl::block_ptr p2_out = p2.outputs()["p2_out"];
    b2->zero_grad();
    p2_out->grad(0,0,0) = 1;

    p2.backward();

    EXPECT_FLOAT_EQ(b2->grad(0,0,0), 1);
    EXPECT_FLOAT_EQ(b2->grad(0,0,1), 0);
    EXPECT_FLOAT_EQ(b2->grad(0,1,0), 0);
    EXPECT_FLOAT_EQ(b2->grad(0,1,1), 0);

}

#endif // NEURAL_LIB_MAXPOOL_TEST_H
