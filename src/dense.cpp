
#include "dense.hpp"

namespace nl {

    Dense::Dense(std::string name, std::string fn_name, Op* op, 
                 uint16_t depth, uint16_t width, uint16_t height):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {

        // check that input block is correct

        // create output block

        // create weight block/tensor

        // create threshold block

    }

    Dense::Dense(std::string name, std::string fn_name, Block* input, 
                 uint16_t depth, uint16_t width, uint16_t height):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {

    }
    
    block_map Dense::outputs() {
        block_map map;

        map.insert(std::pair<std::string, Block*>(output->name,
                                                  output));
        return std::move(map);                                          
    }

    block_map Dense::inputs() {
        block_map map;

        map.insert(std::pair<std::string, Block*>(input->name,
                                                  input));
        return std::move(map);                                                      
    }

} // namespace nl
