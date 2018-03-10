#ifndef NEURAL_LIB_PERCEPTRON_H
#define NEURAL_LIB_PERCEPTRON_H

#include <vector>
#include <unordered_map>

namespace nl {

    class Perceptron : public Op {
    public:
        
        Perceptron(std::string name, std::vector<Block *> inputs):
        name(name) {
            
            // insert inputs to input list

            // create output block

            // create weights and thresholds                        

        }

        Perceptron(std::string name, Block* input):
        name(name) {                        

        }

        template<typename... Args>
        Perceptron(std::string name, Block* input, Args... args):
        Perceptron(name, args...) {
            
        }

        virtual void forward() {}

        virtual void backward() {}

    private:

    };


} // namespace nl

#endif // NEURAL_LIB_PERCEPTRON_H
