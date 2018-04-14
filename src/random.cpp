
#include "random.hpp"

namespace nl {

    std::random_device Generator::gen;
    std::uniform_real_distribution<float> Generator::uniform(-1.0, 1.0);

    void Generator::init_random(block_ptr block) {

        auto dims = block->dimensions();
        
        for (uint16_t i = 0; i < dims[0]; i++) {
            for (uint16_t j = 0; j < dims[1]; j++) {
                for (uint16_t k = 0; k < dims[2]; k++) {
                    block->data(i,j,k) = Generator::get();
                }
            }
        }

    }


}
