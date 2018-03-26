
#ifndef NEURAL_LIB_RANDOM_H
#define NEURAL_LIB_RANDOM_H

#include <random>

#include "block.hpp"

namespace nl {

    // random number generator
    class Generator {
    public:
        static float get() {
            return uniform(gen);
        }
        static std::random_device gen;
        static std::uniform_real_distribution<float> uniform;
        static void init_random(Block* block);
    };
}

#endif // NEURAL_LIB_RANDOM_H
