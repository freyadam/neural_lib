#ifndef NEURAL_LIB_NET_H
#define NEURAL_LIB_NET_H

#include <unordered_map>
#include <vector>
#include <memory>

#include "op.hpp"
#include "block.hpp"

namespace nl {

	// class Net : public Op{	

	// 	std::shared_ptr<block> add_block(std::string name) {
	// 		block_ptr new_block = make_shared<Block>();		   
	// 	}

	// 	std::shared_ptr<op> add_op(std::string name) {

	// 	}

    //     void forward() {} // TODO
    //     void backward() {} // TODO

    //     virtual block_map inputs() { // TOOD
    //         return block_map();
    //     }

    //     virtual block_map outputs() { // TODO
    //         return block_map();
    //     }

	// 	unordered_map<std::string, std::shared_ptr<block>> blocks;
	// 	unordered_map<std::string, std::shared_ptr<op>> ops;

	// };

} // namespace nl

#endif // NEURAL_LIB_NET_H
