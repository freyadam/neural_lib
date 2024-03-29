
#include "error.hpp"

#include <iostream>

namespace nl {

    float Error::L1(const std::vector<block_ptr> & net_outputs, 
                    const std::vector<block_ptr> & correct_outputs) {
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

    float Error::L1(block_ptr net_output,
                    block_ptr correct_output) {

        std::vector<block_ptr> net_vect;
        net_vect.push_back(net_output);

        std::vector<block_ptr> correct_vect;
        net_vect.push_back(correct_output);

        return Error::L1(net_vect, correct_vect);                
    }

    std::vector<Eigen::Tensor<float,3>>
        Error::L1_grad(const std::vector<block_ptr> & net_outputs, 
                       const std::vector<block_ptr> & correct_outputs) {

        if (net_outputs.size() != correct_outputs.size())
            throw InputException();

        std::vector<Eigen::Tensor<float, 3>> ret;

        for (uint16_t i = 0; i < net_outputs.size(); i++) {
            Eigen::Tensor<float,3> t = 
                (net_outputs[i]->data - correct_outputs[i]->data);
            ret.push_back(t/t.abs());
        }

        return ret;                
    }

    Eigen::Tensor<float, 3> 
    Error::L1_grad(block_ptr net_output, 
                   block_ptr correct_output) {

        std::vector<block_ptr> net_vect;
        net_vect.push_back(net_output);

        std::vector<block_ptr> correct_vect;
        net_vect.push_back(correct_output);

        return Error::L1_grad(net_vect, correct_vect)[0];                
    }


    float Error::L2(const std::vector<block_ptr> & net_outputs, 
                    const std::vector<block_ptr> & correct_outputs) {
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

    float Error::L2(block_ptr net_output,                     
                    block_ptr correct_output) {

        std::vector<block_ptr> net_vect;
        net_vect.push_back(net_output);

        std::vector<block_ptr> correct_vect;
        net_vect.push_back(correct_output);

        return Error::L2(net_vect, correct_vect);
    }

    std::vector<Eigen::Tensor<float, 3>>
        Error::L2_grad(const std::vector<block_ptr> & net_outputs, 
                       const std::vector<block_ptr> & correct_outputs) {

        if (net_outputs.size() != correct_outputs.size())
            throw InputException();

        std::vector<Eigen::Tensor<float, 3>> ret;

        for (uint16_t i = 0; i < net_outputs.size(); i++) {
            Eigen::Tensor<float,3> t = 
                (net_outputs[i]->data - correct_outputs[i]->data);
            ret.push_back(t);
        }

        return ret;
    }

    Eigen::Tensor<float, 3> Error::L2_grad(block_ptr net_output,                     
                                           block_ptr correct_output) {

        std::vector<block_ptr> net_vect;
        net_vect.push_back(net_output);

        std::vector<block_ptr> correct_vect;
        net_vect.push_back(correct_output);

        return Error::L2_grad(net_vect, correct_vect)[0];
    }

} // namespace nl
