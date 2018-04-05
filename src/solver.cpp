
#include "solver.hpp"

namespace nl {

    Solver::Solver(Net* net, Block* output_block, Block* desired_block):
        net(net), lr(0.1), 
        nesterov(false), 
        decay_factor(0.1), cycle_length(1000), steps_without_change(0) {
        output.push_back(output_block);
        desired.push_back(desired_block);            
    }

    float Solver::train(uint16_t cycles) {

        for (uint16_t i = 0; i < cycles; ++i) {

            // possibly update learning rate
            if (steps_without_change >= cycle_length)
                lr *= decay_factor;
            steps_without_change++;

            // zero out gradient from previous cycle
            for (auto & block_pair : net->blocks) {
                Block* b = block_pair.second;
                b->zero_grad();
            }
                
            // forward pass
            net->forward();
                
            // compute error gradient on the op output and 
            // save it in the output blocks
            std::vector<Eigen::Tensor<float,3>> grads = 
                Error::L2_grad(output, desired);
            for (uint16_t i = 0; i < output.size(); ++i) {
                output[i]->grad = grads[i];
            }

            // backward
            net->backward();

            // update weights
            for (auto & block_pair : net->blocks) {
                Block* b = block_pair.second;
                if (b->trainable) 
                    b->data -= lr * b->grad;
            }
        }

        return Error::L2(output, desired);
    }

    void Solver::setMethod(std::string name) {
        
        if (name == "sgd")
            nesterov = false;
        else if (name == "nesterov")
            nesterov = true;
        else
            throw InputException();

    }

    void Solver::setLRDecay(float multiplier, uint16_t period) {
        decay_factor = multiplier;
        cycle_length = period;            
    }

} // namespace nl
