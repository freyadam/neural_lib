
#include "solver.hpp"

namespace nl {

    float Solver::train(uint16_t cycles) {

        for (uint16_t i = 0; i < cycles; ++i) {

            // zero out gradient from previous cycle
            for (auto & block_pair : net->blocks) {
                Block* b = block_pair.second;
                b->zero_grad();
            }
                
            // forward pass
            net->forward();
                
            // compute error gradient on the op output and 
            // save it in the output blocks
            // std::vector<Eigen::Tensor<float, 3>> grads = 
            Eigen::Tensor<float,3> grad = 
                Error::L2_grad(output[0], desired[0]);
            // for (uint16_t i = 0; i < output.size(); ++i) {
            //     output[i]->grad = grads[i];
            // }
            output[0]->grad = grad;

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

} // namespace nl
