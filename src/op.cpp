
#include "op.hpp"

namespace nl {

    Op::~Op() {
        // possibly delete owned blocks
        if (delete_owned) 
            for (Block * block : owned) {
                delete block;
            }
    }        

    void Op::zero_grad() {
        // zero out inputs
        for (auto pair : inputs()) {
            Block* b = pair.second;
            b->zero_grad();
        }
        // zero out outputs
        for (auto pair : outputs()) {
            Block* b = pair.second;
            b->zero_grad();
        }
    }

} // namespace nl
