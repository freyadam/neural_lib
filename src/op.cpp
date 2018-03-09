
#include "op.hpp"

namespace nl {

Op::~Op() {
    // possibly delete owned blocks
    if (delete_owned) 
        for (Block * block : owned) {
            delete block;
        }
}        

} // namespace nl
