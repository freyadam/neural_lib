


#include "solver.hpp"

namespace nl {

    Solver::Solver(Net* net, block_ptr output_block, block_ptr desired_block):
        net(net), lr(0.1), 
        nesterov(false), 
        decay_factor(0.1), cycle_length(1000), steps_without_change(0) {
        output.push_back(output_block);
        desired.push_back(desired_block);            
    }

    float Solver::train(uint16_t cycles) {
        
        if (nesterov)
            return train_nesterov(cycles);
        else 
            return train_sgd(cycles);

    }

    float Solver::train_sgd(uint16_t cycles) {

        for (uint16_t i = 0; i < cycles; ++i) {

            // possibly update learning rate
            if (steps_without_change >= cycle_length)
                lr *= decay_factor;
            steps_without_change++;

            // zero out gradient from previous cycle
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
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

            // update weights using gradient
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
                if (b->trainable) 
                    b->data -= lr * b->grad;
            }

        }

        return Error::L2(output, desired);
    }

    float Solver::train_nesterov(uint16_t cycles) {

        for (uint16_t i = 0; i < cycles; ++i) {

            // possibly update learning rate
            if (steps_without_change >= cycle_length)
                lr *= decay_factor;
            steps_without_change++;

            // zero out gradient from previous cycle
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
                b->zero_grad();
            }

            // update weights using momentum term
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
                if (b->trainable) 
                    b->data -= (lr/3) * momentum[b->name];
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

            // update weights using gradient
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
                if (b->trainable) 
                    b->data -= lr * b->grad;
            }

            // update momentum
            for (auto & block_pair : net->blocks) {
                block_ptr b = block_pair.second;
                if (b->trainable) 
                    momentum[b->name] =
                        inertia * momentum[b->name] +
                        (1-inertia) * b->grad;
            }
        }

        return Error::L2(output, desired);
    }

    void Solver::setMethod(std::string name) {
        
        if (name == "sgd")
            nesterov = false;
        else if (name == "nesterov") {            
            nesterov = true;
            // potentially initialize momentum map
            if (momentum.empty())
                init_momentum();
        } else
            throw InputException();

    }

    void Solver::setLRDecay(float multiplier, uint16_t period) {
        decay_factor = multiplier;
        cycle_length = period;            
    }

    void Solver::init_momentum() {

        // create a momentum tensor of correct dimension 
        // for each trainable block
        for (auto & block_pair : net->blocks) {
            block_ptr block = block_pair.second;
            if (block->trainable) {

                // get dimensions
                auto dims = block->dimensions();

                // create new momentum tensor
                momentum.insert(std::pair<std::string, Eigen::Tensor<float,3>>
                                (block->name,
                                 Eigen::Tensor<float,3>(dims[0],
                                                        dims[1],
                                                        dims[2])));
                // zero out the tensor
                momentum[block->name].setZero();
            }
        }

    }

} // namespace nl
