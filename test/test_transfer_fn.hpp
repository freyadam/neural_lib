
#ifndef NEURAL_LIB_TRANSFER_FN_TEST_H
#define NEURAL_LIB_TRANSFER_FN_TEST_H

#include <cmath>

#include "transfer_fns.hpp"

TEST(TransferFnTest, Sigmoid) {

    nl::TransferFn * fn = nl::TransferFns::get("sigmoid");

    // forward
    EXPECT_FLOAT_EQ(fn->forward(0), 0.5);
    EXPECT_TRUE(fn->forward(1) > 0.5);
    EXPECT_TRUE(fn->forward(-1) < 0.5);

    // backward
    EXPECT_FLOAT_EQ(fn->backward(0), 0.25);                
    EXPECT_TRUE(fn->backward(10) > fn->backward(100));
    EXPECT_TRUE(fn->backward(-10) > fn->backward(100));
}

TEST(TransferFnTest, Tanh) {

    nl::TransferFn * fn = nl::TransferFns::get("tanh");

    // forward
    EXPECT_FLOAT_EQ(fn->forward(0), 0);
    EXPECT_TRUE(fn->forward(1) > 0);
    EXPECT_TRUE(fn->forward(-1) < 0);

    // backward
    EXPECT_FLOAT_EQ(fn->backward(0), 1);                
    EXPECT_TRUE(fn->backward(10) > fn->backward(100));
    EXPECT_TRUE(fn->backward(-10) > fn->backward(100));
}

TEST(TransferFnTest, ReLU) {

    nl::TransferFn * fn = nl::TransferFns::get("relu");

    // forward
    EXPECT_FLOAT_EQ(fn->forward(0.1), 0.1);
    EXPECT_FLOAT_EQ(fn->forward(-0.1), 0);

    // backward
    EXPECT_FLOAT_EQ(fn->backward(1), 1);                
    EXPECT_FLOAT_EQ(fn->backward(-1), 0);                
}

TEST(TransferFnTest, Softplus) {

    nl::TransferFn * fn = nl::TransferFns::get("softplus");

    // forward
    EXPECT_FLOAT_EQ(fn->forward(0), std::log(2));
    EXPECT_TRUE(fn->forward(-1) < fn->forward(0));
    EXPECT_TRUE(fn->forward(0) < fn->forward(1));

    // backward
    EXPECT_TRUE(fn->backward(100) > fn->backward(10));                
    EXPECT_TRUE(fn->backward(10) > fn->backward(1));                
    EXPECT_TRUE(fn->backward(1) > fn->backward(-1));                
    EXPECT_TRUE(fn->backward(-1) > fn->backward(-10));                
}

TEST(TransferFnTest, Linear) {

    nl::TransferFn * fn = nl::TransferFns::get("linear");

    // forward
    EXPECT_FLOAT_EQ(fn->forward(0), 0);
    EXPECT_FLOAT_EQ(fn->forward(1), 1);
    EXPECT_FLOAT_EQ(fn->forward(-2), -2);

    // backward
    EXPECT_FLOAT_EQ(fn->backward(-10), 1);
    EXPECT_FLOAT_EQ(fn->backward(-1), 1);
    EXPECT_FLOAT_EQ(fn->backward(0), 1);
    EXPECT_FLOAT_EQ(fn->backward(1), 1);
    EXPECT_FLOAT_EQ(fn->backward(10), 1);
}

#endif // NEURAL_LIB_TRANSFER_FN_TEST_H
