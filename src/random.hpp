
#ifndef NEURAL_LIB_RANDOM_H
#define NEURAL_LIB_RANDOM_H

#include <random>

#include "block.hpp"

namespace nl {

    /// Random number generator wrapper class.
    class Generator {
    public:
        /// Get a single random float from uniform distribution [-1,1]
        static float get() {
            return uniform(gen);
        }
        ///
        /// Fill data tensor of block with random float from uniform
        /// distribution [-1,1]
        /// @param block block which data tensor needs to be generated  
        static void init_random(block_ptr block);
        /// random number generator 
        static std::random_device gen;
        /// Uniform distribution float number generator
        static std::uniform_real_distribution<float> uniform;
    };
}

#endif // NEURAL_LIB_RANDOM_H
