#ifndef NEURAL_LIB_READER_H
#define NEURAL_LIB_READER_H

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>

#include "cimg/CImg.h"
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>

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
        CsvReader(std::string name, std::string file_addr); 

        /// Read a single line and save individual records in output blocks.
        void forward();

        // Reader has no inputs and as such no place to propagate gradient to.
        void backward() {}
        
        virtual block_map inputs();

        virtual block_map outputs();

    private:
        ///
        /// Delimiter for records on a single line. 
        /// Lines are always delimited by '\n'.
        /// 
        static const char delimiter = ',';
        
        /// Output block
        block_ptr output;

        /// Stream for reading individual lines
        std::ifstream line_stream;

        /// csv file address.
        std::string file_addr;

        // default constructor, for serialization
        CsvReader(): Op("default_name") {}

        template<class Archive>
        void save(Archive & ar, const unsigned int) const
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & output;
            ar & file_addr;
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int)
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & output;
            ar & file_addr;
            // start reading from the beginning, state of reader within
            // file is not preserved
            line_stream = std::ifstream(file_addr, std::ifstream::in);
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

        friend class boost::serialization::access;
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
            block_map m = {std::pair<std::string, block_ptr>
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
        block_ptr output_block;

        /// ifstream corresponding to text file with image addresses
        std::ifstream line_stream;

        /// address of text file with image addresses
        std::string file_addr;

        // default constructor, for serialization
        ImgReader(): Op("default_name") {}

        template<class Archive>
        void save(Archive & ar, const unsigned int) const
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & output_block;
            ar & file_addr;
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int)
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & output_block;
            ar & file_addr;
            // start reading from the beginning, state of reader within
            // file is not preserved
            line_stream = std::ifstream(file_addr, std::ifstream::in);
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

        friend class boost::serialization::access;
    };


} // namespace nl

#endif // NEURAL_LIB_READER_H
