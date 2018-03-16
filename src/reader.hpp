#ifndef NEURAL_LIB_READER_H
#define NEURAL_LIB_READER_H

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <unordered_map>

namespace nl {

    class CsvReader : public Op {
    public:
        CsvReader(std::string name, std::string file_addr):
        Op(name), file_addr(file_addr) {
                     
            // open file
            std::ifstream file(file_addr, std::ifstream::in);
            
            /*
              check that all lines have the same number of records 
              and that they are valid floats
            */
            bool first_read = false;
            uint16_t record_count = 0;
            std::string line;
            while (std::getline(file, line)) {
                std::stringstream line_stream(line);
                std::string record;
                                
                if (!first_read) { // set number of records in first line
                    first_read = true;
                    record_count = std::count(line.begin(), line.end(), 
                                              CsvReader::delimiter);

                    // empty lines are not very interesting
                    if (record_count == 0)
                        throw InputException();

                } else { // check that all lines contain the same number of records
                    if (record_count != std::count(line.begin(), line.end(), 
                                                   CsvReader::delimiter))
                        throw InputException();
                }

                // try to read each record into float to see if it 
                // is in valid format
                while(std::getline(line_stream, record, 
                                   CsvReader::delimiter)) {
                    std::stof(record);
                }
            }

            // empty file
            if (!first_read)
                throw InputException();

            // create 'record_count' number of 1x1x1 output blocks
            for (uint16_t i = 0; i < record_count; ++i) {
                Block* rec = new Block(name + "_" + std::to_string(i),
                                       1, 1, 1);
                output_blocks.push_back(rec);
                // output blocks with dumb pointers need to be deleted
                owned.push_back(rec); 
            }

            // initialize stream on the beginning of the file
            line_stream = std::ifstream(file_addr, std::ifstream::in);

        }

        /// Read a single line and save individual records in output blocks.
        void forward() {            
            std::string line;

            // if end of file was reached, start reading 
            if (!std::getline(line_stream, line)) {
                // issue with reading from file, not end of line
                if (!line_stream.eof()) {
                    throw InputException();
                }
                // otherwise start reading again from the beginning
                line_stream = std::ifstream(file_addr, std::ifstream::in);
                std::getline(line_stream, line);
            }

            std::stringstream line_stream(line);
            std::string record;

            // load sequence of records on a single line to output blocks
            uint16_t i = 0;
            while(std::getline(line_stream, record, 
                               CsvReader::delimiter)) {                
                output_blocks[0]->data(0,0,0) = std::stof(record);
                i++;
            }            

        }

        // Reader has no inputs and as such no place to propagate gradient to.
        void backward() {}
        
        virtual block_map inputs() {
            return std::move(block_map());
        }

        virtual block_map outputs() {
            block_map m;

            for (Block* b : output_blocks) {
                m.insert(std::make_pair(b->name, b));
            }

            return std::move(m);
        }

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
        ImgReader(std::string name, std::string file_addr):
        Op(name) {}

        void forward() {

        }

        // reader has no inputs and as such no place to propagate gradient to
        void backward() {}
        
        // reader has no inputs so return empty map
        virtual block_map inputs() {
            return std::move(block_map());
        }

        virtual block_map outputs() {
            return std::move(block_map()); // TODO fix
        }

    };


} // namespace nl

#endif // NEURAL_LIB_READER_H
