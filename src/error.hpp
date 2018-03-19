
#ifndef NEURAL_LIB_ERROR_H
#define NEURAL_LIB_ERROR_H

#include <cmath>
#include <vector>

#include "block.hpp"
#include "exceptions.hpp"

 // TODO rewrite to get rid of duplicity

namespace nl {
    
    /// Class holding several pure functions to compute error.
    class Error {
    public:
        /// L1 norm, note: vectors could be switched without change of result
        /// @param net_outputs output blocks of a network
        /// @param correct_outputs blocks with correct results that should be fitted
        static float L1(std::vector<Block*> net_outputs, 
                        std::vector<Block*> correct_outputs);

        /// L1 norm gradient for each block (in the same order as is the output)
        /// @param net_outputs output blocks of a network
        /// @param correct_outputs blocks with correct results that should be fitted
        static std::vector<float> L1_grad(std::vector<Block*> net_outputs, 
                                          std::vector<Block*> correct_outputs);

        /// L1 norm
        /// @param net_output output block of a network
        /// @param correct_output block with correct results that should be fitted
        static float L1(Block* net_output, Block* correct_output);

        /// L1 norm gradient
        /// @param net_output output block of a network
        /// @param correct_output block with correct results that should be fitted
        static float L1_grad(Block* net_output, Block* correct_output);

        /// L2 norm, note: vectors could be switched without change of result
        /// @param net_outputs output blocks of a network
        /// @param correct_outputs blocks with correct results that should be fitted
        static float L2(std::vector<Block*> net_outputs, 
                        std::vector<Block*> correct_outputs);

        /// L2 norm gradient for each block (in the same order as is the output)
        static std::vector<float> L2_grad(std::vector<Block*> net_outputs, 
                                          std::vector<Block*> correct_outputs);

        /// L2 norm
        /// @param net_output output block of a network
        /// @param correct_output block with correct results that should be fitted
        static float L2(Block* net_output, Block* correct_output);

        /// L2 norm gradient
        /// @param net_output output block of a network
        /// @param correct_output block with correct results that should be fitted
        static float L2_grad(Block* net_output, Block* correct_output);
    };

}

#endif // NEURAL_LIB_ERROR_H
