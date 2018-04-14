#ifndef NEURAL_LIB_OP_H
#define NEURAL_LIB_OP_H

#include <vector> 
#include <unordered_map>

#include "block.hpp"

// resolve circular dependency
class Block;

namespace nl {

    /// Abstract class representing all nodes in computational graph
    /// that actually perform any kind of computation.
	class Op {
    public:

        /// Constructor 
        /// @param name name of the operation
        Op(std::string name): name(name) {}

        /// Destructor.
        virtual ~Op() {}

        ///
        /// Forward pass of the computational operation. Depends on input blocks.
        /// Modifies output blocks. 
        ///
        virtual void forward() = 0;
            
        ///
        /// Backward pass of the computational operation. Propagates gradient to
        /// input blocks.
        ///
        virtual void backward() = 0;

        /// Blocks that are taken into account but not modified during forward pass
        virtual block_map inputs() = 0;

        /// Blocks that are modified during forward pass
        virtual block_map outputs() = 0;

        ///
        /// Set gradients of all neighbouring blocks (both inputs and outputs)
        /// to zero.
        ///
        void zero_grad();

        ///
        /// Name of the operation. It needs to be unique within a network 
        /// as it is used as an identifier,
        ///
		std::string name;        

    private:
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & name;
        }

        friend class boost::serialization::access;
	};

} // namespace nl

#endif // NEURAL_LIB_OP_H
