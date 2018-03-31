#ifndef NEURAL_LIB_SOFTMAX_H
#define NEURAL_LIB_SOFTMAX_H

#include <vector>
#include <unordered_map>
#include <iostream>

#include "random.hpp"
#include "op.hpp"
#include "transfer_fns.hpp"

namespace nl {

    /// 
    /// Softmax class, takes a block of any dimension and outputs a bloc
    /// of the same dimension weighted using the softmax function which
    /// is in more details specified here: 
    /// https://en.wikipedia.org/wiki/Softmax_function
    ///
    class Softmax : public Op {
    public:

        /// Constructor.
        /// @param name name of the softmax op
        /// @param op previous operation with only a single one output block of
        /// correct dimension
        Softmax(std::string name, Op* op); // TODO change to Op*

        /// Constructor.
        /// @param name name of the softmax op
        /// @param input input block
        Softmax(std::string name, Block* input);

        virtual void forward();

        virtual void backward();
    
        virtual block_map outputs();

        virtual block_map inputs();

    private:
        ///
        /// Auxiliary method called by constructor,
        /// creates output block
        /// 
        void create_output(Block* input);
        /// Input block
        Block* input;
        /// Output block
        Block* output;
    };

} // namespace nl

#endif // NEURAL_LIB_SOFTMAX_H
