
#include <iostream>
#include <random>
#include <string> 

#include "neural.hpp"

using namespace std;

/*
Example: XOR learning

Purpose of this example is to demonstrate several features of the library.
Most importantly, the function of a reader, in this case nl::CsvReader, 
and automatic presentation of the input to the network by solver. 
Reader 'r' with access to training samples in file "xor_values.csv" is part
of the network and as such there is no need for manual input of values into
network. 

As reader generates both input for network and desired data, it is
followed by two dense layers with static weights to separate training and 
correct data into two distinct parts. To make them static, trainable flag
is set to false for both weight and threshold.

Separators could in theory be replaced by another reader that would hold 
only desired outputs while the former reader would keep only training
data.

Another two dense layers are created that form the trained network. 
All layers are then added to the network 'net' which is trained by 
'solver'. Order in which layers are added to the network is not important.

As in the previous example, network is trained by 'solver.train()'. 
In this case, parameter of 'train' defines the number of consecutive 
sequences forward_pass->backward_pass->update.

Sidenote: One of the issues of training XOR neural network is that
for some (randomly-generated) original weights, net can reach only 
local maximum. From my testing, this case was quite frequent but
as this training is performed very quickly, I did not consider it
to be an unsurpassable problem as otherwise this problem nicely 
demonstrates features I wanted to show. In my case, probability
of reaching only local maximum in 5 consecutive runs was 
extremely low.
*/

void modify_sep_weights(nl::Dense & sep_corr, nl::Dense & sep_input) {

    // only third value is propagated
    nl::Block& sep_corr_w = *sep_corr.inputs()["correct_w"];
    nl::Block& sep_corr_thr = *sep_corr.inputs()["correct_thr"];
    sep_corr_w.trainable = false;
    sep_corr_w.data(0,0,0) = 0;
    sep_corr_w.data(0,0,1) = 0;
    sep_corr_w.data(0,0,2) = 1;
    sep_corr_thr.trainable = false;
    sep_corr_thr.data(0,0,0) = 0;

    // first value is propagated to first position,
    // second value is propagated to second position
    nl::Block& sep_input_w = *sep_input.inputs()["input_w"];
    nl::Block& sep_input_thr = *sep_input.inputs()["input_thr"];
    sep_input_w.trainable = false;
    sep_input_w.data(0,0,0) = 1;
    sep_input_w.data(0,0,1) = 0;
    sep_input_w.data(0,0,2) = 0;
    sep_input_w.data(0,0,3) = 0;
    sep_input_w.data(0,0,4) = 1;
    sep_input_w.data(0,0,5) = 0;
    sep_input_thr.trainable = false;
    sep_input_thr.data(0,0,0) = 0;
    sep_input_thr.data(0,0,1) = 0;

}

int main(int argc, char *argv[])
{

    // ----- create network components -----
    // reader for that reads inputs and correct outputs from file
    nl::CsvReader r("reader", "xor_values.csv");
    
    // dense layer that only separate the single data input into 
    // two separate blocks
    // output of sep_corr is used as desired output of network
    // output of sep_input is used as input of network
    nl::Dense sep_corr("correct", "linear", r,
                       1, 1, 1);
    nl::Dense sep_input("input", "linear", r,
                        1, 1, 2);
    // modify separator weights to really just divide data into 2 blocks
    modify_sep_weights(sep_corr, sep_input);

    // create network layers
    nl::Dense l1("l1", "relu", sep_input,
                      1, 1, 2);
    nl::Dense l2("l2", "relu", l1,
                      1, 1, 1);

    // ----- create network and put individual ops inside -----
    nl::Net net("net");
    net.add(&r);
    net.add(&sep_corr);
    net.add(&sep_input);
    net.add(&l1);
    net.add(&l2);    

    // ----- create solver that is supposed to train the network -----
    nl::Solver solver(net, net.outputs()["l2_out"], net.outputs()["correct_out"]);
    // solver.setLRDecay(0.3, 1000); // set learning rate decay
    // solver.setMethod("nesterov"); // use training with momentum
   
    // ----- train net for given number of iterations -----
    uint16_t iterations = 30;
    for (uint16_t i = 0; i < iterations; ++i) {

        // run several cycle of the training algorithm
        // error values should get progressively smaller
        // as network is optimized 
        float err = solver.train(10 * 4 + 1);
        std::cout << err << " ----- "
                  << net.outputs()["l2_out"]->data(0,0,0) << "/"
                  << net.outputs()["correct_out"]->data(0,0,0)
                  << std::endl;        
    }

    return 0;
}
