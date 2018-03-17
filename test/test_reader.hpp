
#ifndef NEURAL_LIB_READER_TEST_H
#define NEURAL_LIB_READER_TEST_H

#include "reader.hpp"

// file does not exist
TEST(ReaderTest, NonExistingFile) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "csv/definitely_not_valid_adress.csv"), 
                 nl::InputException);
}

// file is empty 
TEST(ReaderTest, EmptyFile) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "csv/empty_file.csv"), 
                 nl::InputException);
}

// no records in lines
TEST(ReaderTest, NoRecords) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "csv/empty_lines.csv"), 
                 nl::InputException);
}

// valid input that loops
TEST(ReaderTest, Valid) {

    nl::CsvReader r("reader", "csv/valid.csv");
    nl::Block* o1 = r.outputs()["reader_out0"];
    nl::Block* o2 = r.outputs()["reader_out1"];

    // second pass 
    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 1.0);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), 2.3);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 3.12);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), -6);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 2e4);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), -1.23e-1);

    // second pass 
    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 1.0);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), 2.3);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 3.12);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), -6);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 2e4);
    EXPECT_FLOAT_EQ(o2->data(0,0,0), -1.23e-1);
}

#endif // NEURAL_LIB_READER_TEST_H
