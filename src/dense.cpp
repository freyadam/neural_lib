
#include "dense.hpp"

namespace nl {

    Dense::Dense(std::string name, std::string fn_name, Op* op, 
                 uint16_t depth, uint16_t width, uint16_t height):
        Op(name), transfer_fn(TransferFns::get(fn_name)) {

        // check that input op has only a single output
        if (op->outputs().size() != 1)
            throw InputException();

        // get output of op
        input = op->outputs().begin()->second;                

        // 3-dim block
        auto input_dims = input->dimensions();
        if (input_dims.size() == 3)
            throw DimensionException();

        create_blocks(input_dims[0] * input_dims[1] * input_dims[2],
                      depth, width, height);
    }

    Dense::Dense(std::string name, std::string fn_name, Block* input, 
                 uint16_t depth, uint16_t width, uint16_t height):
        Op(name), input(input), transfer_fn(TransferFns::get(fn_name)) {

        // 3-dim block
        auto input_dims = input->dimensions();
        if (input_dims.size() == 3)
            throw DimensionException();

        create_blocks(input_dims[0] * input_dims[1] * input_dims[2],
                      depth, width, height);
    }

    void Dense::create_blocks(uint16_t input_size, uint16_t depth, 
                       uint16_t width, uint16_t height) {

        // create output block
        output = new Block(name + "_out", depth, width, height);
        owned.push_back(output);

        // create weight block
        // individual weight between each cell in input and output blocks
        uint16_t weight_size = 
            input_size
            * depth * width * height;
        weight = new Block(name + "_w", 1, 1, weight_size);
        weight->trainable = true;
        Generator::init_random(weight);
        owned.push_back(weight);

        // create threshold block
        threshold = new Block(name + "_thr", depth, width, height);
        threshold->trainable = true;
        Generator::init_random(threshold);
        owned.push_back(threshold);
        
    }

    void Dense::forward() {

        struct Coord {
            uint16_t x, y, z;
        };
        Coord from; // position in input block
        Coord to; // position in output block

        // dimensions of output block
        uint16_t to_d = output->dimensions()[0];
        uint16_t to_w = output->dimensions()[1];
        uint16_t to_h = output->dimensions()[2];
        // dimensions of input block
        uint16_t from_d = input->dimensions()[0];
        uint16_t from_w = input->dimensions()[1];
        uint16_t from_h = input->dimensions()[2];

        // weights
        Eigen::TensorMap<Eigen::Tensor<float, 6>> w_data(weight->data.data(),
                                                         from_d, from_w, from_h,
                                                         to_d, to_w, to_h);                                     

        // specify position in output block
        for (to.x = 0; to.x < to_d; ++to.x) {
            for (to.y = 0; to.y < to_w; ++to.y) {
                for (to.z = 0; to.z < to_h; ++to.z) {

                    // delete any previous values
                    output->data(to.x, to.y, to.z) = 0;

                    // specify position in input block
                    for (from.x = 0; from.x < from_d; ++from.x) {
                        for (from.y = 0; from.y < from_w; ++from.y) {
                            for (from.z = 0; from.z < from_h; ++from.z) {

                                // add weighted input
                                output->data(to.x, to.y, to.z) +=
                                    w_data(from.x, from.y, from.z,
                                           to.x, to.y, to.z) 
                                    * input->data(from.x, from.y, from.z);                                    
                    
                            }            
                        }            
                    }                               

                    // add threshold
                    output->data(to.x, to.y, to.z) +=
                        threshold->data(to.x, to.y, to.z);

                    // apply transfer function
                    output->data(to.x, to.y, to.z)
                        = transfer_fn.backward(output->data(to.x, to.y, to.z));
                    
                }            
            }            
        }            
        
    }

    void Dense::backward() {
        
        struct Coord {
            uint16_t x, y, z;
        };
        Coord from; // position in input block
        Coord to; // position in output block        

        // dimensions of output block
        uint16_t to_d = output->dimensions()[0];
        uint16_t to_w = output->dimensions()[1];
        uint16_t to_h = output->dimensions()[2];
        // dimensions of input block
        uint16_t from_d = input->dimensions()[0];
        uint16_t from_w = input->dimensions()[1];
        uint16_t from_h = input->dimensions()[2];

        // pass gradient through transfer function
        Eigen::Tensor<float,3> grad(to_d, to_w, to_h);
        for (uint16_t i = 0; i < to_d; ++i) {
            for (uint16_t j = 0; j < to_w; ++j) {
                for (uint16_t k = 0; k < to_h; ++k) {
                    grad(i,j,k) = output->grad(i,j,k) *
                        transfer_fn.backward(output->data(i,j,k));
                }   
            }   
        }

        Eigen::TensorMap<Eigen::Tensor<float, 6>> w_data(weight->data.data(),
                                     from_d, from_w, from_h,
                                     to_d, to_w, to_h);                                     
        Eigen::TensorMap<Eigen::Tensor<float, 6>> w_grad(weight->grad.data(),
                                          from_d, from_w, from_h,
                                          to_d, to_w, to_h);                                     

        // compute input gradient
        for (to.x = 0; to.x < to_d; ++to.x) {
            for (to.y = 0; to.y < to_w; ++to.y) {
                for (to.z = 0; to.z < to_h; ++to.z) {

                    // specify position in input block
                    for (from.x = 0; from.x < from_d; ++from.x) {
                        for (from.y = 0; from.y < from_w; ++from.y) {
                            for (from.z = 0; from.z < from_h; ++from.z) {

                                input->grad(from.x, from.y, from.z) += 
                                    w_data(from.x, from.y, from.z,
                                           to.x, to.y, to.z) *
                                    grad(to.x, to.y, to.z);                                
                                
                            }            
                        }            
                    }                               
                    
                }            
            }            
        }                    
        
        // weight gradient
        // specify position in output block
        for (to.x = 0; to.x < to_d; ++to.x) {
            for (to.y = 0; to.y < to_w; ++to.y) {
                for (to.z = 0; to.z < to_h; ++to.z) {

                    // specify position in input block
                    for (from.x = 0; from.x < from_d; ++from.x) {
                        for (from.y = 0; from.y < from_w; ++from.y) {
                            for (from.z = 0; from.z < from_h; ++from.z) {

                                w_grad(from.x, from.y, from.z,
                                       to.x, to.y, to.z) +=
                                    grad(to.x, to.y, to.z) *
                                    input->data(from.x, from.y, from.z);
                                  
                            }            
                        }            
                    }                               
                    
                }            
            }            
        }                    

        // threshold gradient
        for (uint16_t i = 0; i < to_d; ++i) {
            for (uint16_t j = 0; j < to_w; ++j) {
                for (uint16_t k = 0; k < to_h; ++k) {
                    threshold->data(i,j,k) += grad(i,j,k);
                }   
            }   
        }
   
    }
    
    block_map Dense::outputs() {
        block_map map;

        map.insert(std::pair<std::string, Block*>(output->name,
                                                  output));
        return map;
    }

    block_map Dense::inputs() {
        block_map map;

        map.insert(std::pair<std::string, Block*>(input->name,
                                                  input));
        map.insert(std::pair<std::string, Block*>(weight->name,
                                                  weight));
        map.insert(std::pair<std::string, Block*>(threshold->name,
                                                  threshold));
        return map;
    }

} // namespace nl
