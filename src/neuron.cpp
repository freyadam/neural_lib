
#include "neuron.hpp"

namespace nl {

    Neuron::Neuron(std::string name, std::string fn_name, Op & op):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {                        

        // check that previous operation has exactly a single output block
        if (op.outputs().size() != 1)
            throw InputException();

        block_ptr input = op.outputs().begin()->second;
        
        // check that input is of correct dimension
        auto dims = input->dimensions();
        if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
            dims.size() != 3)
            throw DimensionException();

        // insert block and appropriate weight to vector
        InputPair ip;                
        block_ptr weight = std::make_shared<Block>(name + "_" + input->name + "_w", 1, 1, 1);
        weight->data(0,0,0) = Generator::get();
        weight->trainable = true; // weights should be trained

        ip.input = input;
        ip.weight = weight;
        input_vector.push_back(ip);

        // create threshold block
        threshold = std::make_shared<Block>(name + "_thr", 
                                            1,1,1);

        // set original threshold
        threshold->data(0,0,0) = Generator::get();
        threshold->trainable = true; // weights should be trained

        // create output block
        output = std::make_shared<Block>(name + "_out",
                                         1, 1, 1);
    }

    Neuron::Neuron(std::string name, std::string fn_name, const std::vector<block_ptr> & inputs):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {                        
            
        for (auto & input : inputs) {
            if (input == nullptr)
                throw nl::InputException();
        }

        // process inputs
        for (auto & input : inputs) {

            // check that all inputs are of correct dimension
            auto dims = input->dimensions();
            if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
                dims.size() != 3)
                throw DimensionException();

            // insert block and appropriate weight to vector
            InputPair ip;                
            block_ptr weight = std::make_shared<Block>(name + "_" + input->name + "_w", 1, 1, 1);
            weight->data(0,0,0) = Generator::get();
            weight->trainable = true; // weights should be trained
            ip.input = input;
            ip.weight = weight;
            input_vector.push_back(ip);
        }

        // create threshold block
        threshold = std::make_shared<Block>(name + "_thr", 
                                                     1,1,1);
        // set original threshold
        threshold->data(0,0,0) = Generator::get();
        threshold->trainable = true; // threshold should be trained

        // create output block
        output = std::make_shared<Block>(name + "_out",
                           1, 1, 1);
    }

    Neuron::Neuron(std::string name, std::string fn_name, block_ptr input):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {                        

        if (input == nullptr)
            throw nl::InputException();

        // check that input is of correct dimension
        auto dims = input->dimensions();
        if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
            dims.size() != 3)
            throw DimensionException();

        // insert block and appropriate weight to vector
        InputPair ip;                
        block_ptr weight = std::make_shared<Block>(name + "_" + input->name + "_w", 1, 1, 1);
        weight->data(0,0,0) = Generator::get();
        weight->trainable = true;

        ip.input = input;
        ip.weight = weight;
        input_vector.push_back(ip);

        // create threshold block
        threshold = std::make_shared<Block>(name + "_thr", 
                              1,1,1);
        // set original threshold
        threshold->data(0,0,0) = Generator::get();
        threshold->trainable = true;

        // create output block
        output = std::make_shared<Block>(name + "_out",
                           1, 1, 1);        
    }

    void Neuron::forward() {
        Eigen::Tensor<float, 3> x(1,1,1);
        x(0,0,0) = 0;

        for (auto & p : input_vector) {
            x += p.input->data * p.weight->data;                
        }

        x += threshold->data;

        output->data(0,0,0) = transfer_fn->forward(x(0,0,0));
    }

    void Neuron::backward() {
        Eigen::Tensor<float, 3> grad = 
            output->grad * transfer_fn->backward(output->data(0,0,0));

        for (auto & p : input_vector) {    
            // propagate gradient to input block 
            p.input->grad += grad * p.weight->data;
            // propagate gradient to input block 
            p.weight->grad += grad * p.input->data;
        }
        
        // propagate gradient to threshold block
        threshold->grad += grad;        
    }

    block_map Neuron::outputs() {
        block_map map;

        map.insert(std::pair<std::string, block_ptr>(output->name,
                                                  output));

        return map;
    }

    block_map Neuron::inputs() {
        block_map map;

        // insert inputs 
        for (auto & p : input_vector) {
            map.insert(std::make_pair(p.input->name, p.input));
        }

        // insert weights
        for (auto & p : input_vector) {
            map.insert(std::make_pair(p.weight->name, p.weight));
        }

        // insert threshold
        map.insert(std::make_pair(threshold->name, threshold));

        return map;
    }

} // namespace nl
