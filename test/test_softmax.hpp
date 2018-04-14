
#ifndef NEURAL_LIB_SOFTMAX_TEST_H
#define NEURAL_LIB_SOFTMAX_TEST_H

#include <cmath>

#include "softmax.hpp"

// correct input and output blocks
TEST(SoftmaxTest, IO) {
    
    nl::block_ptr b = std::make_shared<nl::Block>("b", 3, 7, 2);
    nl::Softmax s("smax", b);

    auto in = s.inputs();
    auto out = s.outputs();

    // input of correct size
    EXPECT_EQ(in.size(), 1);
    // input fits the block defined in constructor call
    EXPECT_EQ(in.begin()->second, b);

    // output of correct size
    EXPECT_EQ(out.size(), 1);

    // output block correctly named
    nl::block_ptr b_out = out.begin()->second;
    EXPECT_STREQ(b_out->name.c_str(), "smax_out");

    // output block of correct dim
    auto dim = b_out->dimensions();
    EXPECT_EQ(dim.size(), 3);
    EXPECT_EQ(dim[0], b->dimensions()[0]);
    EXPECT_EQ(dim[1], b->dimensions()[1]);
    EXPECT_EQ(dim[2], b->dimensions()[2]);
}

// correct forward pass
TEST(SoftmaxTest, Forward) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 2, 2);
    nl::Softmax s("smax", b);
    nl::block_ptr out = s.outputs()["smax_out"];

    b->data(0,0,0) = 3;
    b->data(0,0,1) = 0;
    b->data(0,1,0) = 0;
    b->data(0,1,1) = 0.5;

    s.forward();

    EXPECT_FLOAT_EQ(out->data(0,0,0), 
                    std::exp(3) / (std::exp(3) + std::exp(0.5) + 2));
    EXPECT_FLOAT_EQ(out->data(0,1,1), 
                    std::exp(0.5) / (std::exp(3) + std::exp(0.5) + 2));

    b->data(0,0,0) = 0;
    b->data(0,0,1) = 0;
    b->data(0,1,0) = 1;
    b->data(0,1,1) = 0;

    s.forward();

    EXPECT_FLOAT_EQ(out->data(0,0,0), 
                    1 / (std::exp(1) + 3));
    EXPECT_FLOAT_EQ(out->data(0,0,1),
                    1 / (std::exp(1) + 3));
    EXPECT_FLOAT_EQ(out->data(0,1,0),
                    std::exp(1) / (std::exp(1) + 3));
    EXPECT_FLOAT_EQ(out->data(0,1,1), 
                    1 / (std::exp(1) + 3));

    b->data(0,0,0) = 2.7;
    b->data(0,0,1) = 2.7;
    b->data(0,1,0) = 2.7;
    b->data(0,1,1) = 2.7;

    s.forward();

    EXPECT_FLOAT_EQ(out->data(0,0,0), 0.25);
    EXPECT_FLOAT_EQ(out->data(0,0,1), 0.25);
    EXPECT_FLOAT_EQ(out->data(0,1,0), 0.25);
    EXPECT_FLOAT_EQ(out->data(0,1,1), 0.25);
    
}

// correct backward pass
TEST(SoftmaxTest, Backward) {

    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 2, 2);
    nl::Softmax s("smax", b);
    nl::block_ptr out = s.outputs()["smax_out"];

    out->data(0,0,0) = 1;
    out->data(0,0,1) = 0;
    out->data(0,1,0) = 2;
    out->data(0,1,1) = 0;

    out->grad(0,0,0) = 1;
    out->grad(0,0,1) = 1;
    out->grad(0,1,0) = 2;
    out->grad(0,1,1) = 1;
    
    s.backward();

    // dE / db_0,0
    EXPECT_FLOAT_EQ(b->grad(0,0,0), 
                    out->data(0,0,0) * (1 - out->data(0,0,0)) * out->grad(0,0,0) +
                    out->data(0,0,0) * (- out->data(0,0,1)) * out->grad(0,0,1) +
                    out->data(0,0,0) * (- out->data(0,1,0)) * out->grad(0,1,0) +
                    out->data(0,0,0) * (- out->data(0,1,1)) * out->grad(0,1,1));
    
    // dE / db_0,1
    EXPECT_FLOAT_EQ(b->grad(0,0,1), 
                    out->data(0,0,1) * (- out->data(0,0,0)) * out->grad(0,0,0) +
                    out->data(0,0,1) * (1 - out->data(0,0,1)) * out->grad(0,0,1) +
                    out->data(0,0,1) * (- out->data(0,1,0)) * out->grad(0,1,0) +
                    out->data(0,0,1) * (- out->data(0,1,1)) * out->grad(0,1,1));

}

#endif // NEURAL_LIB_SOFTMAX_TEST_H
