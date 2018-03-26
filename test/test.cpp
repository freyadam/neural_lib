
#include "gtest/gtest.h"

#include "test_graph.hpp"
#include "test_block.hpp"
#include "test_neuron.hpp"
#include "test_transfer_fn.hpp"
#include "test_reader.hpp"
#include "test_softmax.hpp"
#include "test_net.hpp"

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
