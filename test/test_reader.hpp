
#ifndef NEURAL_LIB_READER_TEST_H
#define NEURAL_LIB_READER_TEST_H

#include "reader.hpp"

// file does not exist
TEST(CsvReaderTest, NonExistingFile) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "test/csv/definitely_not_valid_adress.csv"), 
                 nl::InputException);
}

// file is empty 
TEST(CsvReaderTest, EmptyFile) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "test/csv/empty_file.csv"), 
                 nl::InputException);
}

// no records in lines
TEST(CsvReaderTest, NoRecords) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "test/csv/empty_lines.csv"), 
                 nl::InputException);
}

// valid input that loops
TEST(CsvReaderTest, Valid) {

    nl::CsvReader r("reader", "test/csv/valid.csv");
    nl::Block* o1 = r.outputs()["reader_out"];

    // second pass 
    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 1.0);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), 2.3);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 3.12);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), -6);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 2e4);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), -1.23e-1);

    // second pass 
    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 1.0);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), 2.3);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 3.12);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), -6);

    r.forward();
    EXPECT_FLOAT_EQ(o1->data(0,0,0), 2e4);
    EXPECT_FLOAT_EQ(o1->data(0,0,1), -1.23e-1);

}

// non-existing file
TEST(ImgReaderTest, NonExistingFile) {
    EXPECT_THROW(nl::CsvReader("reader", 
                               "test/img/definitely_not_valid_adress.csv"), 
                 nl::InputException);
}

// empty list
TEST(ImgReaderTest, EmptyFile) {
    EXPECT_THROW(nl::ImgReader("reader", 
                               "test/img/empty.csv"),
                 nl::InputException);
}

// valid input
TEST(ImgReaderTest, ValidFile) {
    EXPECT_NO_THROW(nl::ImgReader("reader", 
                                  "test/img/valid.csv"));
}

// dimension mismatches
TEST(ImgReaderTest, DimMismatch) {
    EXPECT_THROW(nl::ImgReader("reader", 
                               "test/img/dim_mismatch.csv"),
                 nl::DimensionException);
}

// TODO forward for ImgReader

#endif // NEURAL_LIB_READER_TEST_H
