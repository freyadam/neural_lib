
#include "random.hpp"

std::default_random_engine nl::Generator::gen;
std::uniform_real_distribution<float> nl::Generator::uniform(-1.0, 1.0);

nl::Generator generator;
