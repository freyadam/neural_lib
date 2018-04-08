
#ifndef NEURAL_LIB_MAXPOOL_H
#define NEURAL_LIB_MAXPOOL_H

#include <unordered_map>
#include <limits>
#include <tuple>

#include "block.hpp"
#include "op.hpp"
#include "exceptions.hpp"

namespace nl {
    
    ///
    /// Pooling layer that has a 2-dimensional window from which it extracts its
    /// maximum. Window moves in second and third dimensions (width & height) so
    /// the output block has the same depth as the input block. It is possible 
    /// to  pad input block with min. values on the side to allow higher 
    /// flexibility of output.
    /// 
    class MaxPool : public Op {
    public:

        /// Constructor
        /// @param name name of the operation
        /// @param input block
        /// @param window_size length of the perception rectangle
        /// @param padding_size number of min. values added to input block
        /// 
        MaxPool(std::string name, Block* input,
                uint16_t window_size, uint16_t padding_size=0);        
        
        /// Constructor
        /// @param name name of the operation
        /// @param input operation
        /// @param window_size length of the perception rectangle
        /// @param padding_size number of min. values added to input block
        /// 
        MaxPool(std::string name, Op* op, 
                uint16_t window_size, uint16_t padding_size=0);

        virtual void forward();

        virtual void backward();

        virtual block_map outputs();

        virtual block_map inputs();

    private:
        /// Input block
        Block* input;
        /// Output block
        Block* output;
        ///
        /// Window size
        /// The window is rectangular and two-dimensional so 
        /// the depth of a window is 1.
        ///
        uint16_t window_size;
        ///
        /// Padding size
        /// Input gets padded equally along the borders in width
        /// and height dimension with min. values, here represented by zeros.        
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
        /// Get maximum in window specified by its left upper corner
        /// @param d first coordinate of corner
        /// @param w second coordinate of corner
        /// @param h third coordinate of corner
        float window_max(uint16_t depth, int16_t width, int16_t height);
        /// Get a position in block in which has the specified window the 
        /// highest element
        /// @param d first coordinate of corner
        /// @param w second coordinate of corner
        /// @param h third coordinate of corner
        std::tuple<uint16_t, uint16_t, uint16_t>
        window_max_element(uint16_t d, int16_t w, int16_t h);

        // default constructor, for serialization
        MaxPool(): Op("default_name") {}

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & input;
            ar & output;
            ar & window_size;
            ar & padding_size;
        }
        friend class boost::serialization::access;
    };

} // namespace nl

#endif // NEURAL_LIB_MAXPOOL_H
