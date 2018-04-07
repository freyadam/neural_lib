
#ifndef NEURAL_LIB_NEURON_TEST_H
#define NEURAL_LIB_NEURON_TEST_H

#include <vector>
#include <set>

#include "block.hpp"
#include "neuron.hpp"

// blocks need to be of correct dimension
TEST(NeuronTest, BlocksCorrectDim) {

    nl::Block b1("b1", 2, 1, 1);
    nl::Block b2("b2", 1, 2, 1);
    nl::Block b3("b3", 1, 1, 2);
    
    EXPECT_THROW(nl::Neuron n1("neuron", "relu", &b1), nl::DimensionException);
    EXPECT_THROW(nl::Neuron n2("neuron", "relu", &b2), nl::DimensionException);
    EXPECT_THROW(nl::Neuron n3("neuron", "relu", &b3), nl::DimensionException);
}

// existing output of correct dim
TEST(NeuronTest, OutputCorrectDim) {
    
    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    std::vector<nl::Block*> inputs = {&b1, &b2};

    nl::Neuron n("n", "sigmoid", inputs);

    EXPECT_EQ(n.outputs().size(), 1);
    
    // get output block of neuron
    nl::Block* out = n.outputs()["n_out"];

    EXPECT_EQ(out->dimensions().size(), 3);
    EXPECT_EQ(out->dimensions()[0], 1);
    EXPECT_EQ(out->dimensions()[1], 1);
    EXPECT_EQ(out->dimensions()[2], 1);
}

// correct forward output
TEST(NeuronTest, Forward) {

    
    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n("n", "linear", &b1, &b2);

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 0;
    n.forward();

    float threshold = n.outputs()["n_out"]->data(0,0,0);

    b1.data(0,0,0) = 1;
    b2.data(0,0,0) = 0;
    n.forward();

    float w1 = n.outputs()["n_out"]->data(0,0,0) - threshold;

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 1;
    n.forward();

    float w2 = n.outputs()["n_out"]->data(0,0,0) - threshold;

    b1.data(0,0,0) = 0.3;
    b2.data(0,0,0) = -1.2;
    n.forward();

    // TODO lower precision
    EXPECT_FLOAT_EQ(n.outputs()["n_out"]->data(0,0,0), 
                    0.3 * w1 + -1.2 * w2 + threshold);
}

// correct backward gradient outputs
TEST(NeuronTest, Backward) {
    
    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n("n", "linear", &b1, &b2);

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 0;
    n.forward();

    float threshold = n.outputs()["n_out"]->data(0,0,0);

    b1.data(0,0,0) = 1;
    b2.data(0,0,0) = 0;
    n.forward();

    float w1 = n.outputs()["n_out"]->data(0,0,0) - threshold;

    b1.data(0,0,0) = 0;
    b2.data(0,0,0) = 1;
    n.forward();

    float w2 = n.outputs()["n_out"]->data(0,0,0) - threshold;    

    float grad_on_output = 0.3;

    b1.zero_grad();
    b2.zero_grad();

    n.outputs()["n_out"]->grad(0,0,0) = grad_on_output;
    n.backward();

    EXPECT_NEAR(b1.grad(0,0,0), grad_on_output * w1, 1e-5);
    EXPECT_NEAR(b2.grad(0,0,0), grad_on_output * w2, 1e-5);
}

// correct outputs() 
TEST(NeuronTest, OutputMethodElements) {

    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n("afawgsg", "softplus", &b1, &b2);

    nl::block_map map = n.outputs();

    EXPECT_EQ(map.size(), 1);    
    EXPECT_EQ(map.begin()->second->name, "afawgsg_out");
}

// correct inputs()
TEST(NeuronTest, InputMethodElements) {

    nl::Block b1("b1", 1, 1, 1);
    nl::Block b2("b2", 1, 1, 1);
    nl::Neuron n("vcx", "softplus", &b1, &b2);

    nl::block_map map = n.inputs();

    // two input blocks + their corresponding weights + threshold
    EXPECT_EQ(map.size(), 5);    

    // create set of all input block names
    std::set<std::string> input_names;
    for (auto it = map.begin(); it != map.end(); ++it) {
        input_names.insert(it->first);
    }

    // all elements are unique
    EXPECT_EQ(input_names.size(), 5);    

    // all of these are present
    EXPECT_TRUE(input_names.find("b1") != input_names.end());
    EXPECT_TRUE(input_names.find("b2") != input_names.end());
    EXPECT_TRUE(input_names.find("vcx_b1_w") != input_names.end());
    EXPECT_TRUE(input_names.find("vcx_b2_w") != input_names.end());
    EXPECT_TRUE(input_names.find("vcx_thr") != input_names.end());
}

#endif // NEURAL_LIB_NEURON_TEST_H
