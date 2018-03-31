
#ifndef NEURAL_LIB_DENSE_H
#define NEURAL_LIB_DENSE_H

#include "block.hpp"
#include "op.hpp"
#include "random.hpp"
#include "transfer_fns.hpp"

namespace nl {

    ///
    /// Dense layer, sometimes called also fully-connected layer as each input 
    /// cell and each output cell have an independent weight value
    /// 
    class Dense : public Op {
    public:

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

        virtual void forward();

        virtual void backward();

        virtual block_map outputs();

        virtual block_map inputs();

    private:
        ///
        /// Create output, weight and threshold blocks and 
        /// initialize them properly.
        ///
        void create_blocks(uint16_t input_size, uint16_t depth, 
                           uint16_t width, uint16_t height);
        /// Input block
        Block* input;
        /// Output block
        Block* output;
        ///
        /// Weight block, 6-dimensional Tensor with first three dimensions 
        /// specifying the connection in an input block and the other three
        /// dimensions specifying connection in an output block. Although 
        /// functionally it is really a 6-dim. tensor, it is stored in a 
        /// 3-dimensional Tensor with first two dimensions set to 1, so
        /// it is actually stored as a vector.
        ///        
        Block* weight;
        /// Threshold block
        Block* threshold;
        /// Transfer function reference
        TransferFn* transfer_fn;
    }; 

} // namespace nl

#endif // NEURAL_LIB_DENSE_H
