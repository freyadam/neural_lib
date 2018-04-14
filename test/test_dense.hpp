
#ifndef NEURAL_LIB_DENSE_TEST_H
#define NEURAL_LIB_DENSE_TEST_H

#include "block.hpp"
#include "neuron.hpp"
#include "dense.hpp"

// constructors
TEST(TestDense, Constructors) {
    
    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 2, 2);    
    nl::Dense d1("dense1", "relu", b, 2, 2, 2);
    nl::Dense d2("dense2", "linear", &d1, 1, 1, 1);    
    
    // d1 
    EXPECT_STREQ(d1.name.c_str(), "dense1");

    // d2
    EXPECT_STREQ(d2.name.c_str(), "dense2");


}

// forward 
TEST(TestDense, Forward) {

    nl::block_ptr in = std::make_shared<nl::Block>("in", 1, 1, 3);    
    nl::Dense d1("d1", "relu", in, 1, 1, 2);
    nl::block_ptr out = d1.outputs()["d1_out"]; 
    nl::block_ptr weight = d1.inputs()["d1_w"]; 
    nl::block_ptr thr = d1.inputs()["d1_thr"]; 

    // set weights
    Eigen::TensorMap<Eigen::Tensor<float, 6>> w(weight->data.data(), 
                                                1,1,3,1,1,2);

    w(0,0,0,0,0,0) = 1;
    w(0,0,1,0,0,0) = 2;
    w(0,0,2,0,0,0) = 3;

    w(0,0,0,0,0,1) = 1;
    w(0,0,1,0,0,1) = -5;
    w(0,0,2,0,0,1) = 0;

    // set thresholds
    thr->data(0,0,0) = 13;
    thr->data(0,0,1) = 0;

    // set input block
    in->data(0,0,0) = 1;
    in->data(0,0,1) = 1;
    in->data(0,0,2) = 1;   
    
    d1.forward();

    EXPECT_FLOAT_EQ(out->data(0,0,0), 19);
    EXPECT_FLOAT_EQ(out->data(0,0,1), 0);
}

// backward
TEST(TestDense, Backward) {

    nl::block_ptr in = std::make_shared<nl::Block>("in", 1, 1, 3);    
    nl::Dense d1("d1", "linear", in, 1, 1, 2);
    nl::block_ptr out = d1.outputs()["d1_out"]; 
    nl::block_ptr weight = d1.inputs()["d1_w"]; 
    nl::block_ptr thr = d1.inputs()["d1_thr"]; 

    // set gradient
    out->grad(0,0,0) = 1;
    out->grad(0,0,1) = 0.5;

    // set weights
    Eigen::TensorMap<Eigen::Tensor<float, 6>> w(weight->data.data(), 
                                                1,1,3,1,1,2);
    w(0,0,0,0,0,0) = 1;
    w(0,0,1,0,0,0) = 2;
    w(0,0,2,0,0,0) = 3;

    w(0,0,0,0,0,1) = 1;
    w(0,0,1,0,0,1) = -5;
    w(0,0,2,0,0,1) = 0;

    // set thresholds
    thr->data(0,0,0) = 13;
    thr->data(0,0,1) = 0;

    // set input block
    in->data(0,0,0) = 1;
    in->data(0,0,1) = 1;
    in->data(0,0,2) = 1;   

    d1.backward();

    // check input gradient
    EXPECT_FLOAT_EQ(in->grad(0,0,0), 1 * 1 + 1 * 0.5);
    EXPECT_FLOAT_EQ(in->grad(0,0,1), 2 * 1 + (-5) * 0.5);
    EXPECT_FLOAT_EQ(in->grad(0,0,2), 3 * 1 + 0 * 0.5);

    // check weight gradient
    Eigen::TensorMap<Eigen::Tensor<float, 6>> w_grad(weight->grad.data(), 
                                                1,1,3,1,1,2);
    EXPECT_FLOAT_EQ(w_grad(0,0,0,0,0,0), 1 * 1);
    EXPECT_FLOAT_EQ(w_grad(0,0,1,0,0,0), 1 * 1);
    EXPECT_FLOAT_EQ(w_grad(0,0,2,0,0,0), 1 * 1);
    EXPECT_FLOAT_EQ(w_grad(0,0,0,0,0,1), 1 * 0.5);
    EXPECT_FLOAT_EQ(w_grad(0,0,1,0,0,1), 1 * 0.5);
    EXPECT_FLOAT_EQ(w_grad(0,0,2,0,0,1), 1 * 0.5);

    // check threshold
    EXPECT_FLOAT_EQ(thr->grad(0,0,0), 1);
    EXPECT_FLOAT_EQ(thr->grad(0,0,1), 0.5);

}

// inputs & outputs
TEST(TestDense, InputOutput) {
    
    nl::block_ptr b = std::make_shared<nl::Block>("b", 1, 2, 2);    
    nl::Dense d1("d1", "relu", b, 2, 2, 2);
    nl::Dense d2("d2", "linear", &d1, 1, 1, 1);    
    
    // d1 
    EXPECT_EQ(d1.inputs().size(), 3);
    EXPECT_EQ(d1.outputs().size(), 1);

    nl::block_ptr d1_out = d1.outputs()["d1_out"];
    nl::block_ptr d1_w = d1.inputs()["d1_w"];
    nl::block_ptr d1_thr = d1.inputs()["d1_thr"];

    EXPECT_EQ(d1_out->dimensions()[0], 2);
    EXPECT_EQ(d1_out->dimensions()[1], 2);
    EXPECT_EQ(d1_out->dimensions()[2], 2);    
    EXPECT_EQ(d1_out->trainable, false);    

    EXPECT_EQ(d1_w->dimensions()[0], 1);
    EXPECT_EQ(d1_w->dimensions()[1], 1);
    EXPECT_EQ(d1_w->dimensions()[2], 4 * 8);    
    EXPECT_EQ(d1_w->trainable, true);    
    
    EXPECT_EQ(d1_thr->dimensions()[0], 2);
    EXPECT_EQ(d1_thr->dimensions()[1], 2);
    EXPECT_EQ(d1_thr->dimensions()[2], 2);    
    EXPECT_EQ(d1_thr->trainable, true);    

    // d2
    EXPECT_EQ(d2.inputs().size(), 3);
    EXPECT_EQ(d2.outputs().size(), 1);

    nl::block_ptr d2_out = d2.outputs()["d2_out"];
    nl::block_ptr d2_w = d2.inputs()["d2_w"];
    nl::block_ptr d2_thr = d2.inputs()["d2_thr"];

    EXPECT_EQ(d2_out->dimensions()[0], 1);
    EXPECT_EQ(d2_out->dimensions()[1], 1);
    EXPECT_EQ(d2_out->dimensions()[2], 1);    
    EXPECT_EQ(d2_out->trainable, false);    

    EXPECT_EQ(d2_w->dimensions()[0], 1);
    EXPECT_EQ(d2_w->dimensions()[1], 1);
    EXPECT_EQ(d2_w->dimensions()[2], 8 * 1);    
    EXPECT_EQ(d2_w->trainable, true);    
    
    EXPECT_EQ(d2_thr->dimensions()[0], 1);
    EXPECT_EQ(d2_thr->dimensions()[1], 1);
    EXPECT_EQ(d2_thr->dimensions()[2], 1);    
    EXPECT_EQ(d2_thr->trainable, true);    

}

#endif // NEURAL_LIB_DENSE_TEST_H
