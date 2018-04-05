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
        Solver(Net* net, Block* output_block, Block* desired_block);
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
        /// Specify learning method. Currently either "nesterov"
        /// for Nesterov gradient descent with momentum or "sgd" for
        /// Stochastic gradient descent.
        void setMethod(std::string name);
        /// 
        /// Multiply the learning rate by "multiplier" after each 
        /// "period" iterations. Generally, "multiplier" should be
        /// between 0 and 1. Setting it negative will throw an exception.
        /// Setting it higher than 1 is possible but will not generate good
        /// results.
        /// @param multiplier factor with which is learning rate multiplied
        /// @param period number of iterations after which multiplication is
        /// performed
        ///
        void setLRDecay(float multiplier, uint16_t period);
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
        /// Is Nesterov gradient update used? If not, SGD is used.
        bool nesterov;
        /// Learning rate is multiplied by this number after each cycle
        float decay_factor;
        /// How often is learning rate changed
        uint16_t cycle_length;
        /// Number of gradient updates without learning rate change.
        uint16_t steps_without_change;
    };


} // namespace nl

#endif // NEURAL_LIB_SOLVER_H
