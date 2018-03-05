#ifndef NEURAL_LIB_NET_H
#define NEURAL_LIB_NET_H

#include <unordered_map>
#include <vector>
#include <memory>

#include "op.h"
#include "blocks.h"

namespace nl {

	class Net {	

		std::shared_ptr<block> add_block(std::string name) {
			block_ptr new_block = make_shared<Block>();

			

		}

		std::shared_ptr<op> add_op(std::string name) {

		}

		unordered_map<std::string, std::shared_ptr<block>> blocks;
		unordered_map<std::string, std::shared_ptr<op>> ops;

	};

} // namespace nl

#endif // NEURAL_LIB_NET_H
