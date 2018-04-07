#ifndef NEURAL_LIB_BLOCK_H
#define NEURAL_LIB_BLOCK_H

#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
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
        /// as it is used as an identifier.
        /// 
		std::string name;

        /// Dimension vector of both data and gradient tensor in better 
        /// format than the default Eigen::Tensor<float, 3>::Dimensions.
        /// @return dimension vector 
        std::vector<uint16_t> dimensions() const {
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
private:

        /// Default constructor. Provided primarily for serialization purposes.
        Block(): Block("default_name", 1, 1, 1) {};

        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int) const
        {
            // save name and 'trainable' flag
            ar & name;
            ar & trainable;
            // save dimensions of tensor
            auto dims = data.dimensions();
            ar & dims[0];
            ar & dims[1];
            ar & dims[2];
            // save contents of data tensor
            for (uint16_t i = 0; i < dims[0]; i++) {
                for (uint16_t j = 0; j < dims[1]; j++) {
                    for (uint16_t k = 0; k < dims[2]; k++) {
                        ar & data(i,j,k);
                    }
                }
            }
            // save contents of grad tensor
            for (uint16_t i = 0; i < dims[0]; i++) {
                for (uint16_t j = 0; j < dims[1]; j++) {
                    for (uint16_t k = 0; k < dims[2]; k++) {
                        ar & grad(i,j,k);
                    }
                }
            }
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int)
        {
            // load name and 'trainable' flag
            ar & name;
            ar & trainable;
            // load dimensions of tensor
            uint16_t depth, width, height;
            ar & depth;
            ar & width;
            ar & height;
            // create tensors
            data = Eigen::Tensor<float,3>(depth, width, height);
            grad = Eigen::Tensor<float,3>(depth, width, height);
            // load data to tensors
            for (uint16_t i = 0; i < depth; i++) {
                for (uint16_t j = 0; j < width; j++) {
                    for (uint16_t k = 0; k < height; k++) {
                        ar & data(i,j,k);
                    }
                }
            }
            for (uint16_t i = 0; i < depth; i++) {
                for (uint16_t j = 0; j < width; j++) {
                    for (uint16_t k = 0; k < height; k++) {
                        ar & grad(i,j,k);
                    }
                }
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()
	};
   
    typedef std::unordered_map<std::string, Block*> block_map;    

} // namespace nl

#endif // NEURAL_LIB_BLOCK_H
