
#include "neuron.hpp"

namespace nl {

    Neuron::Neuron(std::string name, std::string fn_name, std::vector<Block *> inputs):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {                        
            
        // process inputs
        for (auto & input : inputs) {

            // check that all inputs are of correct dimension
            auto dims = input->dimensions();
            if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
                dims.size() != 3)
                throw DimensionException();

            // insert block and appropriate weight to vector
            InputPair ip;                
            Block* weight = new Block(input->name + "_w", 1, 1, 1);
            weight->data(0,0,0) = Generator::get();

            ip.input = input;
            ip.weight = weight;
            input_vector.push_back(ip);

            // created weights need to be deleted in the end
            owned.push_back(weight);
        }

        // create threshold block
        threshold = new Block(name + "_thr", 
                              1,1,1);
        // set original threshold
        threshold->data(0,0,0) = Generator::get();
        
        /// output becomes "owned" so it may be properly deleted
        owned.push_back(threshold);

        // create output block
        output = new Block(name + "_out",
                           1, 1, 1);

        /// output becomes "owned" so it may be properly deleted
        owned.push_back(output);           
    }

    Neuron::Neuron(std::string name, std::string fn_name, Block* input):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {                        

        // check that input is of correct dimension
        auto dims = input->dimensions();
        if (dims[0] != 1 || dims[1] != 1 || dims[2] != 1 ||
            dims.size() != 3)
            throw DimensionException();

        // insert block and appropriate weight to vector
        InputPair ip;                
        Block* weight = new Block(input->name + "_w", 1, 1, 1);
        weight->data(0,0,0) = Generator::get();

        ip.input = input;
        ip.weight = weight;
        input_vector.push_back(ip);

        // created weights need to be deleted in the end
        owned.push_back(weight);                        

        // create threshold block
        threshold = new Block(name + "_thr", 
                              1,1,1);
        // set original threshold
        threshold->data(0,0,0) = Generator::get();
        
        /// output becomes "owned" so it may be properly deleted
        owned.push_back(threshold);

        // create output block
        output = new Block(name + "_out",
                           1, 1, 1);
        /// output becomes "owned" so it may be properly deleted
        owned.push_back(output);    
        
    }

    void Neuron::forward() {
        Eigen::Tensor<float, 3> x(1,1,1);
        x(0,0,0) = 0;

        for (auto & p : input_vector) {
            x += p.input->data * p.weight->data;                
        }

        x += threshold->data;

        output->data = x;
        // output->data(0,0,0) = transfer_fn.forward(x); // TODO
    }

    void Neuron::backward() {
        Eigen::Tensor<float, 3> grad = 
            output->gradient * transfer_fn.backward(output->data(0,0,0));
                                           
        for (auto & p : input_vector) {    
            // propagate gradient to input block 
            p.input->gradient += grad * p.weight->data;
            // propagate gradient to input block 
            p.input->gradient += grad * p.input->data;
        }
        
        // propagate gradient to threshold block
        threshold->gradient += grad;        
    }

    block_map Neuron::outputs() {
        block_map map;

        map.insert(std::pair<std::string, Block*>(output->name,
                                                  output));
            
        return std::move(map);                                          
    }

    block_map Neuron::inputs() {
        block_map map;
        return std::move(map);                      
    }

} // namespace nl
