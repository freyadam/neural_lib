#ifndef NEURAL_LIB_BLOCK_H
#define NEURAL_LIB_BLOCK_H

#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/unsupported/CXX11/Tensor>

namespace nl {
    
    ///
    /// A storage for the network. Used for holding data and gradient 
    /// flowing through the network
    ///
	class Block {
    public:

        /// Constructor.
        /// @param name unique identifer of the block
        /// @param depth depth of both 3D tensors, should be 1 for 2D matrices
        /// @param width width of both 3D tensors
        /// @param height height of both 3D tensors
        Block(std::string name, uint16_t depth, 
              uint16_t width, uint16_t height):
            name(name),
            data(depth, width, height),
            grad(depth, width, height) {
            zero_grad();
        }

        ///
        /// Name of the block. Must be unique within a single network
        /// as it is used as a identifier.
        /// 
		const std::string name;

        /// Dimension vector of both data and gradient tensor in better 
        /// format than the default Eigen::Tensor<float, 3>::Dimensions.
        /// @return dimension vector 
        std::vector<uint16_t> dimensions() {
            std::vector<uint16_t> v;
            
            for (auto & d : data.dimensions()) {
                v.push_back(d);
            }
            
            return v;
        }

        /// Fill 'gradient' tensor with just zeros
        void zero_grad() {
            grad.setZero();
        }

        ///
        /// Actual 3D tensor for storage mostly for op results. Although 
        /// it can be modified by hand.
        /// 
        Eigen::Tensor<float,3> data;

        /// 
        /// Storage for gradient flowing backward during backpropagation.
        /// 
        Eigen::Tensor<float,3> grad;

        ///
        /// True iff it is desirable to change the data based 
        /// on the gradient. Set to true for weights,
        /// set to false for blocks holding data passed through net.
        /// 
        bool trainable = false;

	};
   
    typedef std::unordered_map<std::string, Block*> block_map;    

} // namespace nl

#endif // NEURAL_LIB_BLOCK_H
