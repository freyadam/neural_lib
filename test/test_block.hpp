
#ifndef NEURAL_LIB_BLOCK_TEST_H
#define NEURAL_LIB_BLOCK_TEST_H

#include "block.hpp"

TEST(BlockTest, Name) {

    nl::Block b("block", 1, 1, 1);

    EXPECT_EQ(b.name, "block");
}

TEST(BlockTest, Dimensions) {

    nl::Block b("block", 1, 2, 3);
    std::vector<uint16_t> dim = b.dimensions();

    EXPECT_EQ(dim[0], 1);
    EXPECT_EQ(dim[1], 2);
    EXPECT_EQ(dim[2], 3);
}

TEST(BlockTest, DataRW) {

    nl::Block b("block", 1, 1, 1);

    b.data(0,0,0) = 3.2;
    EXPECT_FLOAT_EQ(b.data(0,0,0), 3.2);

    b.grad(0,0,0) = -1.7;
    EXPECT_FLOAT_EQ(b.grad(0,0,0), -1.7);
}

#endif // NEURAL_LIB_BLOCK_TEST_H
