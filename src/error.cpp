
#include "error.hpp"

#include <iostream>

namespace nl {

    float Error::L1(std::vector<Block*> net_outputs, 
                    std::vector<Block*> correct_outputs) {
        float sum = 0.0;

        if (net_outputs.size() != correct_outputs.size())
            throw InputException();
        // compute error for each pair
        for (uint16_t i = 0; i < net_outputs.size(); i++) {
            // subtract one tensor from the other and 
            // square the resulting elements
            Eigen::Tensor<float,3> t = 
                (net_outputs[i]->data - correct_outputs[i]->data).abs();
            // sum up elements of tensor and add square root to complete error
            sum += sqrt(((Eigen::Tensor<float,3>)t.sum())(0));
        }

        return 0.5 * sum;
    }

    float Error::L1(Block* net_output, Block* correct_output) {

        // compute error for each pair
        // subtract one tensor from the other and 
        // square the resulting elements
        Eigen::Tensor<float,3> t = 
            (net_output->data - correct_output->data).abs();
        // sum up elements of tensor and add square root to complete error
        float sum = sqrt(((Eigen::Tensor<float,3>)t.sum())(0));

        return 0.5 * sum;
    }

    // Eigen::Tensor<float, 3> // TODO
    // Error::L1_grad(Block* net_output, Block* correct_output) {
    // }
    float Error::L2(std::vector<Block*> net_outputs, 
                    std::vector<Block*> correct_outputs) {
        float sum = 0.0;

        if (net_outputs.size() != correct_outputs.size())
            throw InputException();

        // compute error for each pair
        for (uint16_t i = 0; i < net_outputs.size(); i++) {
            // subtract one tensor from the other and 
            // square the resulting elements
            Eigen::Tensor<float,3> t = 
                (net_outputs[i]->data - correct_outputs[i]->data).square();
            // sum up elements of tensor and add it to complete error
            Eigen::Tensor<float,0> t2 = t.sum();
            sum += t2();
        }

        return 0.5 * sqrt(sum);
    }    

    std::vector<Eigen::Tensor<float, 3>>
        Error::L2_grad(std::vector<Block*> net_outputs, 
                       std::vector<Block*> correct_outputs) {

        std::vector<Eigen::Tensor<float, 3>> ret;

        for (uint16_t i = 0; i < net_outputs.size(); i++) {
            Eigen::Tensor<float,3> t = 
                (net_outputs[i]->data - correct_outputs[i]->data);
            ret.push_back(t);
        }

        return ret;
    }

    float Error::L2(Block* net_output,                     
                    Block* correct_output) {
        // compute error for each pair
        // subtract one tensor from the other and 
        // square the resulting elements
        Eigen::Tensor<float,3> t = 
            (net_output->data - correct_output->data).square();

        // sum up elements of tensor and use it to compute error
        Eigen::Tensor<float,0> t2 = t.sum();
        float sum = t2();

        return 0.5 * sqrt(sum);
    }

    Eigen::Tensor<float, 3> Error::L2_grad(Block* net_output,                     
                                           Block* correct_output) {
        Eigen::Tensor<float,3> t = 
            (net_output->data - correct_output->data);

        return t;
    }

} // namespace nl
