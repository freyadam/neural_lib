
#include "random.hpp"

std::random_device nl::Generator::gen;
std::uniform_real_distribution<float> nl::Generator::uniform(-1.0, 1.0);
