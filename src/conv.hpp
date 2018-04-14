
#ifndef NEURAL_LIB_CONV_H
#define NEURAL_LIB_CONV_H

#include <vector>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp>

#include "block.hpp"
#include "op.hpp"
#include "random.hpp"
#include "transfer_fns.hpp"

namespace nl {

    ///
    /// Convolutional layer. All output cells depend only on a spatially 
    /// limited part of input block. Additionaly, they all share the same
    /// weight vector.
    ///
    class Conv : public Op {
    public:

        /// Constructor
        /// @param name name of the operation
        /// @param fn_name transfer function name
        /// @param input input block
        /// @param output_depth of output block
        /// @param window_size height and width of kernel
        /// @param padding_size how many layers of zeros should 
        /// be appended to input block
        /// @param stride by how much does kernel move
        Conv(std::string name, std::string fn_name, block_ptr input, 
             uint16_t output_depth,
             uint16_t window_size, uint16_t padding_size=0, uint16_t stride=1);

        /// Constructor
        /// @param name name of the operation
        /// @param fn_name transfer function name
        /// @param op output of this operation is used as input
        /// @param output_depth of output block
        /// @param window_size height and width of kernel
        /// @param padding_size how many layers of zeros should 
        /// be appended to input block
        /// @param stride by how much does kernel move
        Conv(std::string name, std::string fn_name, Op* op, 
             uint16_t output_depth,
             uint16_t window_size, uint16_t padding_size=0, uint16_t stride=1);

        virtual void forward();

        virtual void backward();

        virtual block_map outputs();

        virtual block_map inputs();
    
    private:        
        /// Shared init method
        void init(uint16_t input_depth);
        /// compute weighted sum for single cell specified by its position
        /// @param d coordinate in first dimension
        /// @param w coordinate in second dimension
        /// @param h coordinate in third dimension
        /// @return weighted sum for a given "window"
        float weighted_sum(uint16_t d, uint16_t w, uint16_t h);
        ///
        /// Update the gradient of all weights for given depth slice and 
        /// the gradient of all input cells given single output cell gradient
        /// @param grad gradient of output cell before non-linearity applied
        /// @param x coordinate in first dimension of output cell
        /// @param y coordinate in second dimension of output cell
        /// @param z coordinate in third dimension of output cell
        /// 
        void grad_window_update(float grad, uint16_t x, uint16_t y, uint16_t z);
        /// Input block
        block_ptr input;
        /// Output block
        block_ptr output;
        ///
        /// All weights for a single depth slice, that is cells with 
        /// the same depth.
        ///
        struct WeightPair {
            /// Block of weights identical for all cells in single 
            /// output depth slice, 
            block_ptr kernel;
            /// Blocks of dimension (1,1,1) representing thresholds for indivudual 
            /// depth slices.
            block_ptr threshold;                

            template<class Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                ar & kernel;
                ar & threshold;
            }
            friend class boost::serialization::access;
        };
        /// Weights belonging to output depth slices 
        std::vector<WeightPair> weights;
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
        /// Transfer function
        TransferFn* transfer_fn;

        // default constructor, for serialization
        Conv(): Op("default_name") {}        

        template<class Archive>
        void save(Archive & ar, const unsigned int) const
        {
            ar << boost::serialization::base_object<nl::Op>(*this);
            ar << input;
            ar << output;
            ar << weights;
            ar << window_size;
            ar << padding_size;
            ar << stride;
            ar << transfer_fn;
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int)
        {
            ar >> boost::serialization::base_object<nl::Op>(*this);

            // delete previous content
            weights.clear();

            ar >> input;
            ar >> output;
            ar >> weights;
            ar >> window_size;
            ar >> padding_size;
            ar >> stride;
            ar >> transfer_fn;
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

        friend class boost::serialization::access;
    };

} // namespace nl

#endif // NEURAL_LIB_CONV_H
