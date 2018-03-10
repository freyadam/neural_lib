#ifndef NEURAL_LIB_OP_H
#define NEURAL_LIB_OP_H

#include <vector> 
#include <unordered_map>

#include "block.hpp"

// resolve circular dependency
class Block;

namespace nl {

	class Op {
    public:

        Op() {
            // created blocks will be destroyed in destructor
            delete_owned = true;
        }

        /// Destructor. Among other things, it can delete owned blocks.
        virtual ~Op();

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

        /// 
        /// Set to false if you don't want blocks created by this function 
        /// to be destroyed in op's destructor.
        ///
        void delete_created_blocks(bool del) {
            delete_owned = del;
        }

        /// Blocks that are taken into account but not modified during forward pass
        std::unordered_map<std::string, Block *> inputs;
        /// Blocks that are modified during forward pass
        std::unordered_map<std::string, Block *> outputs;

    private:
        ///
        /// Name of the operation. It needs to be unique within a network 
        /// as it is used as an identifier,
        ///
		std::string name;        
        /// Blocks that were created during construction of this op. 
        std::vector<Block *> owned;
        /// decides if owned block will deleted in destructor
        bool delete_owned;
	};

} // namespace nl

#endif // NEURAL_LIB_OP_H
