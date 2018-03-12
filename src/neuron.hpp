#ifndef NEURAL_LIB_PERCEPTRON_H
#define NEURAL_LIB_PERCEPTRON_H

#include <vector>
#include <unordered_map>

namespace nl {

    class Neuron : public Op {
    public:
        
        Neuron(std::string name, std::vector<Block *> inputs):
        Op(name) {

            // check that all inputs are distinct
            
            // insert inputs to input list
            for (auto & input : inputs) {

                // check that all inputs are of correct dimension
                auto dims = input->dimensions();
                if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1)
                    throw DimensionException();

                previous.push_back(input);
            }

            // create output block
            output = new Block(name + "_out",
                                      1, 1, 1);

            /// output becomes "owned" so it may be properly deleted
            owned.push_back(output);
           
            // create weights and thresholds                        
            
        }

        Neuron(std::string name, Block* input):
        Op(name) {                        

        }

        template<typename... Args>
        Neuron(std::string name, Block* input, Args... args):
        Neuron(name, args...) {
            
        }

        virtual void forward() {}

        virtual void backward() {}

        virtual std::unordered_map<std::string, Block *> outputs() {
            std::unordered_map<std::string, Block*> map;

            map.insert(std::pair<std::string, Block*>(output->name,
                                                      output));
            
            return std::move(map);                                          
        }

        virtual std::unordered_map<std::string, Block *> inputs() {
            std::unordered_map<std::string, Block*> map;
            return std::move(map);                      
        }

    private:

        /// Block in which result of Neuron::forward() is stored.
        Block* output;
        /// Input blocks.
        std::vector<Block*> previous;
        ///
        /// Weight blocks applied to all inputs. 
        /// Same order of inputs as the "previous" vector. So weights[0] 
        /// corresponds to previous[0], weights[1] to previous[1] etc.
        /// 
        std::vector<Block*> weights;
        /// Threshold value.
        float threshold;        

    };


} // namespace nl

#endif // NEURAL_LIB_PERCEPTRON_H
