
#ifndef NEURAL_LIB_DENSE_H
#define NEURAL_LIB_DENSE_H

#include "block.hpp"
#include "op.hpp"
#include "transfer_fns.hpp"

namespace nl {

    ///
    /// Dense layer, sometimes called also fully-connected layer as each input 
    /// cell and each output cell have an independent weight value
    /// 
    class Dense : public Op {

        /// Constructor.
        /// @param name name of the resulting dense layer
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param op previous operation with only a single one output block of
        /// correct dimension
        /// @param depth depth of output block
        /// @param width width of output block
        /// @param height height of output block
        Dense(std::string name, std::string fn_name, Op* op, 
              uint16_t depth, uint16_t width, uint16_t height);

        /// Constructor.
        /// @param name name of the resulting dense layer
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param input input block
        /// @param depth depth of output block
        /// @param width width of output block
        /// @param height height of output block
        Dense(std::string name, std::string fn_name, Block* input, 
              uint16_t depth, uint16_t width, uint16_t height);

        virtual void forward() {}

        virtual void backward() {}

        virtual block_map outputs();

        virtual block_map inputs();

    private:
        /// Input block
        Block* input;
        /// Output block
        Block* output;

        /// Transfer function reference
        TransferFn& transfer_fn;
    }; 

} // namespace nl

#endif // NEURAL_LIB_DENSE_H
