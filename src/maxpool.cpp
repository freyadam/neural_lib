
#include "maxpool.hpp"

namespace nl {    

    MaxPool::MaxPool(std::string name, block_ptr input,
                     uint16_t window_size, uint16_t padding_size):
        Op(name), input(input), 
        window_size(window_size), padding_size(padding_size){
            
        if (input == nullptr)
            throw nl::InputException();

        auto input_dims = input->dimensions();
        if (input_dims.size() != 3)
            throw nl::DimensionException();

        // there is no reason to generate max from all zeros
        if (padding_size >= window_size)
            throw nl::InputException();

        // window is greater than observed area in at least one dim.
        if (window_size > input_dims[1] + 2 * padding_size ||
            window_size > input_dims[2] + 2 * padding_size)
            throw nl::InputException();            

        // create output
        output = std::make_shared<Block>(name + "_out",
                           input_dims[0], // same depth as input block
                           input_dims[1] + 2 * padding_size - (window_size - 1),
                           input_dims[2] + 2 * padding_size - (window_size - 1));

    }

    MaxPool::MaxPool(std::string name, Op* op, 
                     uint16_t window_size, uint16_t padding_size):
        Op(name), window_size(window_size), padding_size(padding_size){

        if (op == nullptr)
            throw nl::InputException();

        if (op->outputs().size() != 1)
            throw nl::InputException();

        // take the only output of op as input of this max pool layer
        input = op->outputs().begin()->second;
            
        if (input == nullptr)
            throw nl::InputException();

        auto input_dims = input->dimensions();
        if (input_dims.size() != 3)
            throw nl::DimensionException();

        // there is no reason to generate max from all zeros
        if (padding_size >= window_size)
            throw nl::InputException();

        // window is greater than observed area in at least one dim.
        if (window_size > input_dims[1] + 2 * padding_size ||
            window_size > input_dims[2] + 2 * padding_size)
            throw nl::InputException();            

        // create output
        output = std::make_shared<Block>(name + "_out",
                           input_dims[0],
                           input_dims[1] + 2 * padding_size - (window_size - 1),
                           input_dims[2] + 2 * padding_size - (window_size - 1));

    }

    void MaxPool::forward() {

        // specify cell in output block that is being computed.
        for (uint16_t x = 0; x < output->dimensions()[0]; ++x) {
            for (uint16_t y = 0; y < output->dimensions()[1]; ++y) {
                for (uint16_t z = 0; z < output->dimensions()[2]; ++z) {
                    
                    // get max for window for given output cell
                    output->data(x,y,z) = 
                         window_max(x,y - padding_size, z - padding_size);

                }
            }
        }

    }

    void MaxPool::backward() {

        // specify cell in output block
        for (uint16_t x = 0; x < output->dimensions()[0]; ++x) {
            for (uint16_t y = 0; y < output->dimensions()[1]; ++y) {
                for (uint16_t z = 0; z < output->dimensions()[2]; ++z) {

                    // get first input cell that correspond to value 
                    // in output cell
                    auto coords = window_max_element(x,
                                                     y - padding_size,
                                                     z - padding_size);                    
                    // add up gradient of that cell
                    input->grad(std::get<0>(coords),
                                std::get<1>(coords),
                                std::get<2>(coords)) 
                        += output->grad(x,y,z);
                    
                }
            }
        }

    }

    block_map MaxPool::outputs() {
        block_map map;
        map.insert(std::pair<std::string, block_ptr>(output->name,
                                                  output));
        return map;   
    }

    block_map MaxPool::inputs() {
        block_map map;
        map.insert(std::pair<std::string, block_ptr>(input->name,
                                                  input));
        return map;
    }

    float MaxPool::window_max(uint16_t d, int16_t w, 
                              int16_t h) {
        float current = std::numeric_limits<float>::lowest();

        int16_t x = d;
        for (int16_t y = w; y < w + window_size; ++y) {
            for (int16_t z = h; z < h + window_size; ++z) {

                if (y >= 0 && z >= 0)
                    current = std::max(current, 
                                       input->data(x,y,z));

            }
        }

        return current;
    }

    std::tuple<uint16_t, uint16_t, uint16_t>
    MaxPool::window_max_element(uint16_t d, int16_t w, 
                                int16_t h) {

        float prev, current = std::numeric_limits<float>::lowest();            
        int w_max = 0, h_max = 0; 

        int16_t x = d;
        for (int16_t y = w; y < w + window_size; ++y) {
            for (int16_t z = h; z < h + window_size; ++z) {
                
                if (y >= 0 && z >= 0) {
                    prev = current;
                    current = std::max(current, 
                                       input->data(x,y,z));
                    // update coordinates if changer occured
                    if (prev != current) {
                        w_max = y;
                        h_max = z;
                    }                       
                }

            }
        }

        return std::make_tuple(d, w_max, h_max);
    }

} // namespace nl
