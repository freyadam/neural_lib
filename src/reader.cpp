
#include "reader.hpp"

namespace nl {

    CsvReader::CsvReader(std::string name, std::string file_addr):
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
                record_count = 1 + std::count(line.begin(), line.end(), 
                                              CsvReader::delimiter);

                // empty lines are not very interesting
                if (line.find_first_not_of(" ") == std::string::npos) 
                    throw InputException();

            } else { // check that all lines contain the same number of records
                if (record_count != std::count(line.begin(), line.end(), 
                                               CsvReader::delimiter) + 1) 
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

        // create output block of size 1x1xrecord_count
        output = new Block(name + "_out", 1, 1, record_count);
        owned.push_back(output); 

        // initialize stream to the beginning of the file
        line_stream = std::ifstream(file_addr, std::ifstream::in);

    }

    void CsvReader::forward() {            
        std::string line;

        if (!std::getline(line_stream, line)) {
            // issue with reading from file, not end of file
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
            output->data(0,0,i) = std::stof(record);
            i++;
        }            

    }
    
    block_map CsvReader::inputs() {
        return block_map();
    }

    block_map CsvReader::outputs() {
        block_map m;

        m.insert(std::make_pair(output->name, output));

        return m;
    }

    ImgReader::ImgReader(std::string name, std::string file_addr):
        Op(name), file_addr(file_addr) {
            
        // check that text file exists and open it
        std::ifstream file(file_addr, std::ifstream::in);

        // each line contains a valid filename of an image
        // all images must be of the same size
        bool first = false;
        uint16_t channel_no = 0, width = 0, height = 0;
        std::string img_address;
        while (std::getline(file, img_address)) {
            cimg_library::CImg<float> img(img_address.c_str());                

            if (!first) {
                first = true;
                channel_no = img.spectrum();
                width = img.width();
                height = img.height();
            } else {
                if (channel_no != img.spectrum() ||
                    width != img.width() ||
                    height != img.height())
                    throw DimensionException(); 
            }                
        }            

        // empty list
        if (!first)
            throw InputException();

        // create output block            
        output_block = new Block(name + "_out", channel_no, width, height);
        owned.push_back(output_block);

        // piece of code fixing boost::serialization bug/strange_edge_case
        forward();

        // set beginning position in text file
        line_stream = std::ifstream(file_addr, std::ifstream::in);
    }

    void ImgReader::forward() {            
        cimg_library::CImg<float> img(next_image_addr().c_str());                

        // load data in tensor
        for (uint16_t s = 0; s < img.spectrum(); ++s) {
            for (uint16_t w = 0; w < img.width(); ++w) {
                for (uint16_t h = 0; h < img.height(); ++h) {
                    output_block->data(s, w, h) = img(w, h, s);
                }
            }
        }
    }

    std::string ImgReader::next_image_addr() {
        std::string line;

        if (!std::getline(line_stream, line)) {
            // issue with reading from file, not end of line
            if (!line_stream.eof()) {
                throw InputException();
            }
            // otherwise start reading again from the beginning
            line_stream = std::ifstream(file_addr, std::ifstream::in);
            std::getline(line_stream, line);
        }
            
        return line;
    }


} // namespace nl
