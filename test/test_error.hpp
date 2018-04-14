#ifndef NEURAL_LIB_ERROR_TEST_H
#define NEURAL_LIB_ERROR_TEST_H

#include <cmath>

// L1 norm
TEST(ErrorTest, L1Norm) {

    nl::block_ptr b1 = std::make_shared<nl::Block>("b1", 1, 1, 2);
    nl::block_ptr b2 = std::make_shared<nl::Block>("b2", 1, 1, 2);

    // set b1
    b1->data(0,0) = 0;
    b1->data(0,1) = 1;    

    // set b2
    b2->data(0,1) = -1;    
    b2->data(0,1) = 0.5;    

    std::vector<block_ptr*> v1, v2;
    v1.push_back(b1);
    v2.push_back(b2);

    EXPECT_FLOAT_EQ(nl::Error::L1(v1, v2), 1.5);

    nl::block_ptr b3 = std::make_shared<nl::Block>("b3", 1, 1, 1);
    nl::block_ptr b4 = std::make_shared<nl::Block>("b4", 1, 1, 1);

    // set b3
    b3->data(0,0) = 0;

    // set b4
    b4->data(0,0) = -0.3;    

    v1.push_back(b3);
    v1.push_back(b4);

    EXPECT_FLOAT_EQ(nl::Error::L1(v1, v2), 1.5+0.3);

}

// L2 norm
TEST(ErrorTest, L2Norm) {
    
    nl::block_ptr b1 = std::make_shared<nl::Block>("b1", 1, 1, 2);
    nl::block_ptr b2 = std::make_shared<nl::Block>("b2", 1, 1, 2);

    // set b1
    b1->data(0,0) = 0;
    b1->data(0,1) = 1;    

    // set b2
    b2->data(0,1) = -1;    
    b2->data(0,1) = 0.5;    

    std::vector<block_ptr*> v1, v2;
    v1.push_back(b1);
    v2.push_back(b2);

    EXPECT_FLOAT_EQ(nl::Error::L1(v1, v2), sqrt(1.25));

    nl::block_ptr b3 = std::make_shared<nl::Block>("b3", 1, 1, 1);
    nl::block_ptr b4 = std::make_shared<nl::Block>("b4", 1, 1, 1);

    // set b3
    b3->data(0,0) = 0;

    // set b4
    b4->data(0,0) = -0.3;    

    v1.push_back(b3);
    v1.push_back(b4);

    EXPECT_FLOAT_EQ(nl::Error::L1(v1, v2), sqrt(1.25+0.09));    

}

#endif // NEURAL_LIB_ERROR_TEST_H
