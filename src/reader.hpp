#ifndef NEURAL_LIB_READER_H
#define NEURAL_LIB_READER_H

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>

#include "cimg/CImg.h"

#include "block.hpp"
#include "op.hpp"
#include "exceptions.hpp"

namespace nl {

    class CsvReader : public Op {
    public:
        /// Constructor 
        /// @param name name of the Op
        /// @param file_addr address of the csv file
        CsvReader(std::string name, std::string file_addr); //TODO all values into a single 1x1xK vector?

        /// Read a single line and save individual records in output blocks.
        void forward();

        // Reader has no inputs and as such no place to propagate gradient to.
        void backward() {}
        
        virtual block_map inputs() {
            return std::move(block_map());
        }

        virtual block_map outputs();

    private:
        ///
        /// Delimiter for records on a single line. 
        /// Lines are always delimited by '\n'.
        /// 
        static const char delimiter = ',';
        
        /// Vector of output blocks.
        std::vector<Block*> output_blocks;

        /// Stream for reading individual lines
        std::ifstream line_stream;

        /// csv file address.
        std::string file_addr;
    };

    class ImgReader : public Op {
    public:
        // TODO black and white?

        /// Constructor
        /// @param name name of the Op
        /// @param file_addr address of text file with image addresses,
        /// each image address is on its own line
        ImgReader(std::string name, std::string file_addr):
        Op(name) {
            
            // check that text file exists and open it
            std::ifstream file(file_addr, std::ifstream::in);

            // each line contains a valid filename of an image
            // all images must be of the same size
            std::string line;
            while (std::getline(file, line)) {

                
                
            }


            // create output block            
            output_block = new Block(name + "_out", 1, 1, 1); // TODO change dims
            owned.push_back(output_block);

            // set beginning position in text file
            
        }

        void forward() {

        }

        // reader has no inputs and as such no place to propagate gradient to
        void backward() {}
        
        // reader has no inputs so return empty map
        virtual block_map inputs() {
            return std::move(block_map());
        }

        virtual block_map outputs() {
            block_map m = {std::pair<std::string, Block*>
                           (output_block->name, output_block)};
            
            return std::move(m); 
        }

    private:
        Block* output_block;
    };


} // namespace nl

#endif // NEURAL_LIB_READER_H
