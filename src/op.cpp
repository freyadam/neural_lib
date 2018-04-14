
#include "op.hpp"

namespace nl {

    void Op::zero_grad() {
        // zero out inputs
        for (auto pair : inputs()) {
            block_ptr b = pair.second;
            b->zero_grad();
        }
        // zero out outputs
        for (auto pair : outputs()) {
            block_ptr b = pair.second;
            b->zero_grad();
        }
    }

} // namespace nl
