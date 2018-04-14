
#include "softmax.hpp"

namespace nl {

   Softmax:: Softmax(std::string name, Op* op):
        Op(name) {

        // check that previous operation has exactly a single output block
        if (op->outputs().size() != 1)
            throw InputException();

        input = op->outputs().begin()->second;
        create_output(input);
    }

    Softmax::Softmax(std::string name, block_ptr input):
        Op(name), input(input) {
        create_output(input);
    }

    void Softmax::create_output(block_ptr input) {

        // check dimensions
        auto dims = input->dimensions();
        if (dims.size() != 3)            
            throw DimensionException();

        // create output block
        output = std::make_shared<nl::Block>(name + "_out", dims[0], dims[1], dims[2]);

    }

    void Softmax::forward() {

        Eigen::Tensor<float, 3> exp_in = input->data.exp();
        Eigen::Tensor<float, 0> sum = exp_in.sum();

        output->data = exp_in / sum();

    }

    void Softmax::backward() {

        auto dim = input->dimensions();

        Eigen::Tensor<float, 3> t(dim[0], dim[1], dim[2]);
        t.setZero();
        // get first element in tensor, or rather its position
        for (uint16_t i_1 = 0; i_1 < dim[0]; i_1++) {
            for (uint16_t j_1 = 0; j_1 < dim[1]; j_1++) {
                for (uint16_t k_1 = 0; k_1 < dim[2]; k_1++) {
                    // get second element in tensor, or rather its position
                    for (uint16_t i_2 = 0; i_2 < dim[0]; i_2++) {
                        for (uint16_t j_2 = 0; j_2 < dim[1]; j_2++) {
                            for (uint16_t k_2 = 0; k_2 < dim[2]; k_2++) {

                                t(i_1, j_1, k_1) -=                                     
                                    output->data(i_1, j_1, k_1) 
                                    * output->data(i_2, j_2, k_2)
                                    * output->grad(i_2, j_2, k_2);

                                if (i_1 == i_2 && j_1 == j_2 && k_1 == k_2)
                                    t(i_1, j_1, k_1) +=                                    
                                        output->data(i_2, j_2, k_2) 
                                        * output->grad(i_2, j_2, k_2);

                            }
                        }
                    }

                }
            }
        }

        input->grad = t;
    }

    block_map Softmax::outputs() {
        block_map map;

        map.insert(std::pair<std::string, block_ptr>(output->name,
                                                  output));
        return map;
    }

    block_map Softmax::inputs() {
        block_map map;

        map.insert(std::pair<std::string, block_ptr>(input->name,
                                                  input));
        return map;
    }


} // namespace nl
