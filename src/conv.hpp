
#ifndef NEURAL_LIB_MAXPOOL_H
#define NEURAL_LIB_MAXPOOL_H

namespace nl {

    class Conv : public Op {
    public:
    
        virtual void forward();

        virtual void backward();

        virtual block_map outputs();

        virtual block_map inputs();
    
    private:        
        /// Input block
        Block* input;
        /// Output block
        Block* output;
        /// Block of weights identical to all cell in output block.
        Block* kernel;
        /// Block of dimension (1,1,1) representing threshold for neuron.
        Block* threshold;
        ///
        /// Window size
        /// The window is rectangular and two-dimensional so 
        /// the depth of a window is 1.
        ///
        uint16_t window_size;
        ///
        /// Padding size
        /// Input gets padded equally along the borders in width
        /// and height dimension with zeros.        
        /// So the input for a block of dim. (1,2,2) with padding_size == 2 
        /// will look like this:
        /// 000000
        /// 000000
        /// 00xx00
        /// 00xx00
        /// 000000
        /// 000000
        /// 
        uint16_t padding_size;
        ///
        /// How much does the window move after each iteration. Identical for
        /// both dimensions of movement. 
        /// 
        uint16_t stride;
    };

} // namespace nl

#endif // NEURAL_LIB_MAXPOOL_H
