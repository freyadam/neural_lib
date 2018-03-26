#ifndef NEURAL_LIB_SOLVER_H
#define NEURAL_LIB_SOLVER_H

#include <vector>

#include "block.hpp"
#include "net.hpp"
#include "error.hpp"

namespace nl {

    /// Solver tasked with optimizing weights of given network using
    /// supervised learning.
    class Solver {
    public:
        /// Constructor.
        /// @param net neural network that is being trained
        /// @param output_block block in which computation result will
        /// be stored
        /// @param desired_block block with correct result data
        Solver(Net* net, Block* output_block, Block* desired_block):
            net(net), lr(0.1) {
            output.push_back(output_block);
            desired.push_back(desired_block);            
        }
        /// Constructor.
        /// @param net neural network that is being trained
        /// @param output vector of blocks in which computation result will
        /// be storedx
        /// @param desired vector of blocks with correct result data.
        Solver(Net* net, 
               std::vector<Block*> output, 
               std::vector<Block*> desired):
            net(net), output(output), desired(desired), lr(0.1) {}        
        /// Forward pass, backward pass and subsequent weight update
        /// constitutes a single cycle. 
        /// @param cycles how many times should the update be performed
        /// @return error from the last cycle
        float train(uint16_t cycles=1);
    private:
        ///
        /// Network that is being 
        /// trained by the solver.
        ///
        Net* net;    
        /// op outputs
        std::vector<Block*> output;
        /// desired outputs
        std::vector<Block*> desired;
        /// learning rate
        float lr;
        /// momentum learning rate
        float momentum_lr;
    };


} // namespace nl

#endif // NEURAL_LIB_SOLVER_H
