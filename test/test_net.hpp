
#include <vector>

#include "op.hpp"
#include "neuron.hpp"
#include "net.hpp"

// add_op exception
TEST(NetTest, AddOp) {

    nl::Net net("net");

    EXPECT_THROW(net.add(&net), nl::InputException);

    nl::Block b("b", 1, 1, 1);
    nl::Neuron n1("n", "relu", &b);
    nl::Neuron n2("n", "relu", &b);

    net.add(&n1);
    EXPECT_THROW(net.add(&n2), nl::DuplicityException);
}

// inputs() & outputs()
TEST(NetTest, InputsOutputs) {

    nl::Net net("net");

   EXPECT_EQ(net.inputs().size(), 0);
   EXPECT_EQ(net.outputs().size(), 0);

    nl::Block b1("b1", 1, 1, 1);
    nl::Neuron n1("n1", "relu", &b1);
    nl::Neuron n2("n2", "relu", &n1);

    net.add(&n1);
    net.add(&n2);

    // b1 + weight and threshold blocks for n1 and n2
    EXPECT_EQ(net.inputs().size(), 5);
    // output of n2, "n2_out"
    EXPECT_EQ(net.outputs().size(), 1);

    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n3("n3", "relu", &b2);

    net.add(&n3);

    EXPECT_EQ(net.inputs().size(), 8);
    EXPECT_EQ(net.outputs().size(), 2);    
}

TEST(NetTest, Ordering) {

    nl::Net net("net");

    // empty net has trivially no ordering
    EXPECT_EQ(net.get_ordering().size(), 0);

    nl::Block b1("b1", 1, 1, 1);
    nl::Neuron n1("n1", "relu", &b1);
    nl::Neuron n2("n2", "relu", &n1);

    net.add(&n1);
    net.add(&n2);

    // n1 needs to precede n2
    EXPECT_EQ(net.get_ordering().size(), 2);
    EXPECT_STREQ(net.get_ordering()[0]->name.c_str(), "n1");
    EXPECT_STREQ(net.get_ordering()[1]->name.c_str(), "n2");

    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n3("n3", "relu", &b2);

    net.add(&n3);
    std::vector<nl::Op*> ord = net.get_ordering();
    
    // n1 needs to precede n2, otherwise everything is valid
    EXPECT_EQ(ord.size(), 3);
    EXPECT_TRUE(ord[0]->name == "n1" ||
                ord[0]->name == "n3");
    EXPECT_TRUE(ord[1]->name == "n2" ||
                ord[1]->name == "n3");

}
