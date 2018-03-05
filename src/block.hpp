#ifndef NEURAL_LIB_BLOCK_H
#define NEURAL_LIB_BLOCK_H

#include <memory>

#include "op.h"

namespace nl {

	class Block {

		std::string name;


	};

	typedef std::shared_ptr<Block> block_ptr;

} // namespace nl

#endif // NEURAL_LIB_BLOCK_H
