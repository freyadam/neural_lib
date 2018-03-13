
#ifndef NEURAL_LIB_RANDOM_H
#define NEURAL_LIB_RANDOM_H

#include <random>

namespace nl {

    // random number generator
    class Generator {
    public:
        static float get() {
            return uniform(gen);
        }
    private:
        static std::default_random_engine gen;
        static std::uniform_real_distribution<float> uniform;
    };
}

#endif // NEURAL_LIB_RANDOM_H
