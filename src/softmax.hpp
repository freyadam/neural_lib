#ifndef NEURAL_LIB_SOFTMAX_H
#define NEURAL_LIB_SOFTMAX_H

#include <vector>
#include <unordered_map>
#include <iostream>

#include <boost/serialization/shared_ptr.hpp>

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
        Softmax(std::string name, Op & op);

        /// Constructor.
        /// @param name name of the softmax op
        /// @param input input block
        Softmax(std::string name, block_ptr input);

        virtual void forward();

        virtual void backward();
    
        virtual block_map outputs();

        virtual block_map inputs();

    private:
        ///
        /// Auxiliary method called by constructor,
        /// creates output block
        /// 
        void create_output(block_ptr input);
        /// Input block
        block_ptr input;
        /// Output block
        block_ptr output;
    };

} // namespace nl

#endif // NEURAL_LIB_SOFTMAX_H
