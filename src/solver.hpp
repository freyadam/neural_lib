#ifndef NEURAL_LIB_SOLVER_H
#define NEURAL_LIB_SOLVER_H

#include <vector>

namespace nl {

    class Solver {
    public:
        // Solver(Op* op, Op* correct) {};
        Solver(Op* op, Block* block):
            op(op) {
            desired._push_back(block);            
        }
        Solver(Op* op, std::vector<Block*> blocks):
            op(op), desired(blocks) {}

        float train(uint16_t cycles=1) {

            for (uint16_t i = 0; i < cycles; ++i) {
                // zero out gradient
                
                // forward pass

                // compute error gradient on the op output and 
                // save it in the output blocks

                // backward

                // update weights
            }

            return Error::L2(output, desired);
        }

        float train_until_convergence() {
            return 0.0; // TODO
        }
    private:
        ///
        /// Operation (most probably network) that is being 
        /// trained by the solver.
        ///
        Op* op;
        
        /// op outputs
        std::vector<Block*> output;

        /// desired outputs
        std::vector<Block*> desired;

    };


} // namespace nl

#endif // NEURAL_LIB_SOLVER_H
