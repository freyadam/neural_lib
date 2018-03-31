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

    ///
    /// Reader that loads data into its own block(s) from a specified
    /// csv file. Single forward() call reads and stores a single line
    /// from file. When end of file is reached, reader starts again from
    /// the beginning.
    /// 
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
            return block_map();
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

    /// 
    /// Reader that loads images to its block of its own creation.
    /// Images are specified in a text file where each line should contain
    /// exactly one image address. When reader reaches end of text file,
    /// it starts again from the beginning.
    ///
    class ImgReader : public Op {
    public:
        /// Constructor
        /// @param name name of the Op
        /// @param file_addr address of text file with image addresses,
        /// each image address is on its own line
        ImgReader(std::string name, std::string file_addr);

        void forward();

        // reader has no inputs and as such no place to propagate gradient to
        void backward() {}
        
        // reader has no inputs so return empty map
        virtual block_map inputs() {
            return block_map();
        }

        virtual block_map outputs() {
            block_map m = {std::pair<std::string, Block*>
                           (output_block->name, output_block)};            
            return m;
        }

    private:

        ///
        /// return next image address in list, 
        /// if end of file was reached start again from the beginning
        /// 
        std::string next_image_addr();

        /// block in which resulting image is saved
        Block* output_block;

        /// ifstream corresponding to text file with image addresses
        std::ifstream line_stream;

        /// address of text file with image addresses
        std::string file_addr;
    };


} // namespace nl

#endif // NEURAL_LIB_READER_H
