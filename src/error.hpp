
#ifndef NEURAL_LIB_ERROR_H
#define NEURAL_LIB_ERROR_H

#include <cmath>
#include <vector>

namespace nl {
    
    /// Class holding several pure functions to compute error.
    class Error {
    public:
        /// L1 norm, note: vectors could be switched without change of result
        /// @param net_outputs output blocks of a network
        /// @param correct_outputs blocks with correct results that should be fitted
        static float L1(std::vector<Block*> net_outputs, 
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

            return sum;
        }
        
        /// L2 norm, note: vectors could be switched without change of result
        /// @param net_outputs output blocks of a network
        /// @param correct_outputs blocks with correct results that should be fitted
        static float L2(std::vector<Block*> net_outputs, 
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
                sum += ((Eigen::Tensor<float,3>)t.sum())(0);
            }

            return sqrt(sum);
        }
    };

}

#endif // NEURAL_LIB_ERROR_H
