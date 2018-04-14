
#include "conv.hpp"

namespace nl {

    Conv::Conv(std::string name, std::string fn_name, block_ptr input, 
               uint16_t output_depth,
               uint16_t window_size, uint16_t padding_size, uint16_t stride):
        Op(name), input(input),
        window_size(window_size), padding_size(padding_size), stride(stride), 
        transfer_fn(TransferFns::get(fn_name)) {

        init(output_depth);

    }

    Conv::Conv(std::string name, std::string fn_name, Op & op, 
               uint16_t output_depth,
               uint16_t window_size, uint16_t padding_size, uint16_t stride):
        Op(name), 
        window_size(window_size), padding_size(padding_size), stride(stride), 
        transfer_fn(TransferFns::get(fn_name)) {
        
        if (op.outputs().size() != 1)
            throw nl::InputException();

        input = op.outputs().begin()->second;
        init(output_depth);

    }

    void Conv::init(uint16_t output_depth) {

        if (input == nullptr)
            throw nl::InputException();

        auto input_dims = input->dimensions();
        if (input_dims.size() != 3)
            throw nl::DimensionException();

        // there is no reason to generate max from all zeros
        if (padding_size >= window_size)
            throw nl::InputException();

        uint16_t padded_width = input_dims[1] + 2 * padding_size;
        uint16_t padded_height = input_dims[2] + 2 * padding_size;

        // window is greater than observed area in at least one dim.
        if (window_size > padded_width ||
            window_size > padded_height)
            throw nl::InputException();            

        // input depth slice would not be covered symmetricaly 
        // by input windows due to stride
        if ((padded_width - window_size) % stride != 0 ||
            (padded_height - window_size) % stride != 0) 
            throw nl::InputException();            

        // number of strides taken + 1
        uint16_t output_width = ((padded_width - window_size) / stride) + 1;
        uint16_t output_height = ((padded_height - window_size) / stride) + 1;

        // create output block
        output = std::make_shared<Block>(name + "_out", 
                                         output_depth,
                                         output_width,
                                         output_height);

        // create weights 
        for (uint16_t d = 0; d < output_depth; d++) {
            WeightPair p;
            p.kernel = std::make_shared<Block>(name + "_w" 
                                               + std::to_string(d),
                                               input_dims[0],
                                               window_size,
                                               window_size);
            Generator::init_random(p.kernel);
            p.kernel->trainable = true;

            p.threshold = std::make_shared<Block>(name + "_thr" 
                                                  + std::to_string(d),
                                                  1,1,1);
            Generator::init_random(p.threshold);
            p.threshold->trainable = true;

            // push new weight pair to vector
            weights.push_back(p);
        }

    }

    void Conv::forward() { 

        // specify cell in output block that is being computed.
        for (uint16_t x = 0; x < output->dimensions()[0]; ++x) {

            WeightPair p = weights[x];

            for (uint16_t y = 0; y < output->dimensions()[1]; ++y) {
                for (uint16_t z = 0; z < output->dimensions()[2]; ++z) {
                    float result = 0;
                    // compute weighted sum
                    result = weighted_sum(x,y,z);
                    // add threshold
                    result += p.threshold->data(0,0,0);
                    // apply transfer function
                    output->data(x,y,z) = transfer_fn->forward(result);
                }
            }
        }

    }

    void Conv::backward() { 

        // propagate gradient for each output cell
        for (uint16_t x = 0; x < output->dimensions()[0]; ++x) {

            WeightPair p = weights[x];

            for (uint16_t y = 0; y < output->dimensions()[1]; ++y) {
                for (uint16_t z = 0; z < output->dimensions()[2]; ++z) {

                    float grad = output->grad(x,y,z) * 
                        transfer_fn->backward(output->data(x,y,z));
                    
                    grad_window_update(grad, x, y, z);

                    // update threshold gradient
                    p.threshold->grad(0,0,0) += grad;

                }
            }
        }

    }

    block_map Conv::outputs() {
        block_map map;
        map.insert(std::pair<std::string, block_ptr>(output->name,
                                                  output));
        return map;   
    }

    block_map Conv::inputs() {
        block_map map;
        map.insert(std::pair<std::string, block_ptr>(input->name,
                                                  input));
        for (auto & p : weights) {
            auto & kernel = p.kernel;
            auto & threshold = p.threshold;
            map.insert(std::pair<std::string, block_ptr>(kernel->name,
                                                      kernel));
            map.insert(std::pair<std::string, block_ptr>(threshold->name,
                                                      threshold));
        }

        return map; 
    }

    float Conv::weighted_sum(uint16_t d, uint16_t w, uint16_t h) {
    
        block_ptr kernel = weights[d].kernel;

        // specify upper left corner of window in input block
        uint16_t i_y = stride * w;
        uint16_t i_z = stride * h;

        float sum = 0;
        
        for (uint16_t x = 0; x < input->dimensions()[0]; ++x) {
            for (uint16_t y = 0; y < window_size; ++y) {
                for (uint16_t z = 0; z < window_size; ++z) {    

                    // skip if position is outside of input tensor
                    if (i_y + y - padding_size < 0 ||
                        i_z + z - padding_size < 0 ||
                        i_y + y - padding_size >= input->dimensions()[1] ||
                        i_y + y - padding_size >= input->dimensions()[2])
                        continue;

                    sum +=
                        kernel->data(x,y,z) *
                        input->data(x, 
                                    i_y + y - padding_size,
                                    i_z + z - padding_size);
                }
            }
        }

        return sum;
    
    }

    void Conv::grad_window_update(float grad, uint16_t d, uint16_t w, uint16_t h) {

        block_ptr kernel = weights[d].kernel;

        // specify upper left corner of window in input block
        uint16_t i_y = stride * w;
        uint16_t i_z = stride * h;
        
        for (uint16_t x = 0; x < input->dimensions()[0]; ++x) {
            for (uint16_t y = 0; y < window_size; ++y) {
                for (uint16_t z = 0; z < window_size; ++z) {    

                    // skip if position is outside of input tensor
                    if (i_y + y - padding_size < 0 ||
                        i_z + z - padding_size < 0 ||
                        i_y + y - padding_size >= input->dimensions()[1] ||
                        i_y + y - padding_size >= input->dimensions()[2])
                        continue;

                    float weight = kernel->data(x,y,z);
                    float signal = input->data(x, 
                                               i_y + y - padding_size,
                                               i_z + z - padding_size);

                    kernel->grad(x,y,z)
                        += grad * signal;
                    input->grad(x, i_y + y - padding_size, i_z + z - padding_size) 
                        += grad * weight;
                }
            }
        }

    }

} // namespace nl
