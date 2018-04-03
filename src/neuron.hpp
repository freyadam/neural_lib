#ifndef NEURAL_LIB_NEURON_H
#define NEURAL_LIB_NEURON_H

#include <vector>
#include <unordered_map>
#include <iostream>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>

#include "random.hpp"
#include "op.hpp"
#include "transfer_fns.hpp"

namespace nl {

    ///
    /// Operation representing single artificial neuron. Each input and its 
    /// corresponding weight is stored in its own block. Number of inputs is not
    /// specified. Threshold (with its own block) is also added to the potential.
    /// Potential is then passed through transfer function.
    ///
    class Neuron : public Op {
    public:

        /// Constructor.
        /// @param name name of the resulting neuron
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param inputs vector of input blocks
        Neuron(std::string name, std::string fn_name, std::vector<Block *> inputs);

        /// Constructor.
        /// @param name name of the resulting neuron
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param op previous operation with only a single one output block of
        /// correct dimension
        Neuron(std::string name, std::string fn_name, Op* op);

        /// Constructor.
        /// @param name name of the resulting neuron
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param input input block
        Neuron(std::string name, std::string fn_name, Block* input);

        /// Constructor.
        /// @param name name of the resulting neuron
        /// @param fn_name name of used transfer function as defined in TransferFns
        /// @param input input block
        /// @param args rest of the blocks
        template<typename... Args>
        Neuron(std::string name, std::string fn_name, Block* input, Args... args):
            Neuron(name, fn_name, args...) {

            if (input == nullptr)
            throw nl::InputException();

            // check that input is of correct dimension
            auto dims = input->dimensions();
            if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
                dims.size() != 3)
                throw DimensionException();

            // insert block and appropriate weight to vector
            InputPair ip;                
            Block* weight = new Block(name + "_" + input->name + "_w", 1, 1, 1);
            weight->data(0,0,0) = Generator::get();
            weight->trainable = true; // weights should be trained

            ip.input = input;
            ip.weight = weight;
            input_vector.push_back(ip);

            // created weights need to be deleted in the end
            owned.push_back(weight);                        
        }

        virtual void forward();

        virtual void backward();

        virtual block_map outputs();

        virtual block_map inputs();

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & input_vector;
            ar & output;
            ar & threshold;
            // ar & transfer_fn;   // TODO serialize transfer functions
        }

    private:

        /// Input block and weight pair
        struct InputPair {
            Block* input; /// input from previous layer
            Block* weight; /// corresponding weight

            template<class Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                ar & input;
                ar & weight;
            }

            friend class boost::serialization::access;
        };
        /// All pairs input/weight used in this neuron
        std::vector<InputPair> input_vector;        

        /// Block in which result of Neuron::forward() is stored.
        Block* output;
        /// Threshold value.
        Block* threshold;        
        /// Transfer function reference
        TransferFn* transfer_fn;

        friend class boost::serialization::access;
    };


} // namespace nl

#endif // NEURAL_LIB_NEURON_H
