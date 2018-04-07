
#ifndef NEURAL_LIB_CONV_TEST_H
#define NEURAL_LIB_CONV_TEST_H

#include "conv.hpp"

TEST(ConvTest, Constructor) {

    nl::Op* op_ptr = NULL;
    EXPECT_THROW(nl::Conv("conv", "relu", op_ptr, 
                          1, 2),
                 nl::InputException);    

    nl::Block* block_ptr = NULL;
    EXPECT_THROW(nl::Conv("conv", "relu", block_ptr, 
                          1, 2),
                 nl::InputException);    

    nl::Block b("b", 1, 3, 2);
    nl::Block b2("b2", 1, 4, 4);
    // window greater than observed area
    EXPECT_THROW(nl::Conv("conv", "relu", &b, 
                          1, 3),
                 nl::InputException);    
    EXPECT_NO_THROW(nl::Conv("conv", "relu", &b, 
                          1, 3, 1));    
    EXPECT_THROW(nl::Conv("conv", "relu", &b, 
                          1, 5, 1),
                 nl::InputException);    

    // depth slice not covered symmetrically    
    EXPECT_THROW(nl::Conv("conv", "relu", &b, 
                          1, 2, 0, 2),
                 nl::InputException);    
    EXPECT_NO_THROW(nl::Conv("conv", "relu", &b2, 
                          1, 2, 0, 2));    
    

}

TEST(ConvTest, InputOutput) {
    
    nl::Block b("b", 2, 2, 2);
    nl::Conv c1("c1", "linear", &b,
                1, 1);
    nl::Conv c2("c2", "linear", &b,
                3, 2);

    // input correct size
    EXPECT_EQ(c1.inputs().size(), 1 + 1 * 2);
    EXPECT_EQ(c2.inputs().size(), 1 + 3 * 2);

    // output correct size
    EXPECT_EQ(c1.outputs().size(), 1);
    EXPECT_EQ(c2.outputs().size(), 1);

    auto dim1 = c1.outputs()["c1_out"]->dimensions();
    auto dim2 = c2.outputs()["c2_out"]->dimensions();

    // output block dims
    EXPECT_EQ(dim1[0], 1);
    EXPECT_EQ(dim1[1], 2);
    EXPECT_EQ(dim1[2], 2);

    EXPECT_EQ(dim2[0], 3);
    EXPECT_EQ(dim2[1], 1);
    EXPECT_EQ(dim2[2], 1);

}

TEST(ConvTest, Forward1) {

    nl::Block b1("b1", 1, 2, 2);
    nl::Conv c1("c1", "relu", &b1, 1, 2);

    nl::Block& out1 = *c1.outputs()["c1_out"];
    nl::Block& w1 = *c1.inputs()["c1_w0"];
    nl::Block& thr1 = *c1.inputs()["c1_thr0"];    
    
    // set data
    b1.data(0,0,0) = 3;
    b1.data(0,0,1) = 1;
    b1.data(0,1,0) = -2;
    b1.data(0,1,1) = 3;

    // set weights
    w1.data(0,0,0) = 1;
    w1.data(0,0,1) = 2;
    w1.data(0,1,0) = 3;
    w1.data(0,1,1) = 4;

    // set threshold
    thr1.data(0,0,0) = 2;

    c1.forward();

    // positive x so relu(x) == x
    EXPECT_FLOAT_EQ(out1.data(0,0,0), 3 + 2 + (-6) + 12 + 2);                  
    
    // set threshold
    thr1.data(0,0,0) = -200;

    c1.forward();

    // negative x so relu(x) == 0
    EXPECT_FLOAT_EQ(out1.data(0,0,0), 0);                  

}

// stride = 2
TEST(ConvTest, Forward2) {

    nl::Block b("b", 1, 3, 3);
    nl::Conv c("c", "linear", &b, 2, 1, 0, 2);

    nl::Block& out = *c.outputs()["c_out"];
    nl::Block& w1 = *c.inputs()["c_w0"];
    nl::Block& thr1 = *c.inputs()["c_thr0"];    
    nl::Block& w2 = *c.inputs()["c_w1"];
    nl::Block& thr2 = *c.inputs()["c_thr1"];    
    
    // set data
    b.data(0,0,0) = 3;
    b.data(0,0,2) = 1;
    b.data(0,2,0) = -2;
    b.data(0,2,2) = 3;

    // set weights
    w1.data(0,0,0) = 1;
    w2.data(0,0,0) = -1;

    // set thresholds
    thr1.data(0,0,0) = 2;
    thr2.data(0,0,0) = 30;

    c.forward();

    // first depth slice
    EXPECT_FLOAT_EQ(out.data(0,0,0), 3 + 2);                  
    EXPECT_FLOAT_EQ(out.data(0,0,1), 1 + 2);                  
    EXPECT_FLOAT_EQ(out.data(0,1,0), (-2) + 2);                  
    EXPECT_FLOAT_EQ(out.data(0,1,1), 3 + 2);                      

    // second depth slice
    EXPECT_FLOAT_EQ(out.data(1,0,0), (-3) + 30);                  
    EXPECT_FLOAT_EQ(out.data(1,0,1), (-1) + 30);                  
    EXPECT_FLOAT_EQ(out.data(1,1,0), 2 + 30);                  
    EXPECT_FLOAT_EQ(out.data(1,1,1), (-3) + 30);                      

}

// padding = 1
TEST(ConvTest, Forward3) {

    nl::Block b("b", 1, 1, 1);
    nl::Conv c("c", "linear", &b, 1, 2, 1);

    nl::Block& out = *c.outputs()["c_out"];
    nl::Block& w = *c.inputs()["c_w0"];
    nl::Block& thr = *c.inputs()["c_thr0"];    
    
    // set data
    b.data(0,0,0) = 1;

    // set weights
    w.data(0,0,0) = 1;
    w.data(0,0,1) = 2;
    w.data(0,1,0) = 3;
    w.data(0,1,1) = 4;

    // set threshold
    thr.data(0,0,0) = 10;

    c.forward();

    EXPECT_FLOAT_EQ(out.data(0,0,0), 4 + 10);
    EXPECT_FLOAT_EQ(out.data(0,0,1), 3 + 10);
    EXPECT_FLOAT_EQ(out.data(0,1,0), 2 + 10);
    EXPECT_FLOAT_EQ(out.data(0,1,1), 1 + 10);
    
}

TEST(ConvTest, Backward1) {

    nl::Block b1("b1", 1, 2, 2);
    nl::Conv c1("c1", "linear", &b1, 1, 2);

    nl::Block& out1 = *c1.outputs()["c1_out"];
    nl::Block& w1 = *c1.inputs()["c1_w0"];
    nl::Block& thr1 = *c1.inputs()["c1_thr0"];    

    b1.zero_grad();
    w1.zero_grad();
    thr1.zero_grad();    

    // set data
    b1.data(0,0,0) = 3;
    b1.data(0,0,1) = 1;
    b1.data(0,1,0) = -2;
    b1.data(0,1,1) = 3;

    // set weights
    w1.data(0,0,0) = 1;
    w1.data(0,0,1) = 2;
    w1.data(0,1,0) = 3;
    w1.data(0,1,1) = 4;

    // set threshold
    thr1.data(0,0,0) = 2;

    // set gradient on output
    out1.grad(0,0,0) = 2.4;

    c1.forward();
    c1.backward();
    
    // check data gradient
    EXPECT_FLOAT_EQ(b1.grad(0,0,0), 2.4 * 1);
    EXPECT_FLOAT_EQ(b1.grad(0,0,1), 2.4 * 2);
    EXPECT_FLOAT_EQ(b1.grad(0,1,0), 2.4 * 3);
    EXPECT_FLOAT_EQ(b1.grad(0,1,1), 2.4 * 4);

    // check weights gradient
    EXPECT_FLOAT_EQ(w1.grad(0,0,0), 2.4 * 3);
    EXPECT_FLOAT_EQ(w1.grad(0,0,1), 2.4 * 1);
    EXPECT_FLOAT_EQ(w1.grad(0,1,0), 2.4 * -2);
    EXPECT_FLOAT_EQ(w1.grad(0,1,1), 2.4 * 3);

    // check threshold gradient
    EXPECT_FLOAT_EQ(thr1.grad(0,0,0), 2.4);

}

// stride = 2
TEST(ConvTest, Backward2) {

    nl::Block b("b", 1, 3, 3);
    nl::Conv c("c", "linear", &b, 1, 1, 0, 2);

    nl::Block& out = *c.outputs()["c_out"];
    nl::Block& w = *c.inputs()["c_w0"];
    nl::Block& thr = *c.inputs()["c_thr0"];    
    
    // set data
    b.data(0,0,0) = 3;
    b.data(0,0,2) = 1;
    b.data(0,2,0) = -2;
    b.data(0,2,2) = 3;

    // set weights
    w.data(0,0,0) = 1;

    // set thresholds
    thr.data(0,0,0) = 2;

    // set output gradient
    out.grad(0,0,0) = 1;
    out.grad(0,0,1) = 2;
    out.grad(0,1,0) = 3;
    out.grad(0,1,1) = 4;

    b.zero_grad();
    w.zero_grad();
    thr.zero_grad();
    
    c.forward();
    c.backward();

    // check data gradient
    EXPECT_FLOAT_EQ(b.grad(0,0,0), 1 * 1);
    EXPECT_FLOAT_EQ(b.grad(0,0,2), 1 * 2);
    EXPECT_FLOAT_EQ(b.grad(0,2,0), 1 * 3);
    EXPECT_FLOAT_EQ(b.grad(0,2,2), 1 * 4);
    EXPECT_FLOAT_EQ(b.grad(0,1,2), 0);
    EXPECT_FLOAT_EQ(b.grad(0,0,1), 0);

    // check weights gradient
    EXPECT_FLOAT_EQ(w.grad(0,0,0), 
                    1 * 3 + 2 * 1 + 3 * (-2) + 4 * 3);

    // check threshold gradient
    EXPECT_FLOAT_EQ(thr.grad(0,0,0), 1 + 2 + 3 + 4);    

}

// padding = 1
TEST(ConvTest, Backward3) {

    nl::Block b("b", 1, 1, 1);
    nl::Conv c("c", "linear", &b, 1, 2, 1);

    nl::Block& out = *c.outputs()["c_out"];
    nl::Block& w = *c.inputs()["c_w0"];
    nl::Block& thr = *c.inputs()["c_thr0"];    
    
    // set data
    b.data(0,0,0) = 1;

    // set weights
    w.data(0,0,0) = 1;
    w.data(0,0,1) = 2;
    w.data(0,1,0) = 3;
    w.data(0,1,1) = 4;

    // set threshold
    thr.data(0,0,0) = 10;

    // set gradient on output
    out.grad(0,0,0) = 5;
    out.grad(0,0,1) = 6;
    out.grad(0,1,0) = 7;
    out.grad(0,1,1) = 8;

    b.zero_grad();
    w.zero_grad();
    thr.zero_grad();

    c.forward();
    c.backward();
    
    // check data gradient
    EXPECT_FLOAT_EQ(b.grad(0,0,0), 
                    5 * 4 + 6 * 3 + 7 * 2 + 8 * 1);

    // check weight gradient
    EXPECT_FLOAT_EQ(w.grad(0,0,0), 8);
    EXPECT_FLOAT_EQ(w.grad(0,0,1), 7);
    EXPECT_FLOAT_EQ(w.grad(0,1,0), 6);
    EXPECT_FLOAT_EQ(w.grad(0,1,1), 5);

    // check threshold gradient
    EXPECT_FLOAT_EQ(thr.grad(0,0,0), 5 + 6 + 7 + 8);    
    
}

#endif // NEURAL_LIB_CONV_TEST_H
