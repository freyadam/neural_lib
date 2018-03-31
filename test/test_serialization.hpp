
#ifndef NEURAL_LIB_SERIALIZATION_TEST_H
#define NEURAL_LIB_SERIALIZATION_TEST_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

TEST(SerializationTest, Block) {

    // create block
    nl::Block b("b", 1, 1, 2);
    b.trainable = false;
    b.data(0,0,0) = 3.14;
    b.data(0,0,1) = 12e-3;
    b.grad(0,0,0) = -1;
    b.grad(0,0,1) = 3e12;

    // save block
    std::string filename = "test.txt";
    {
        std::ofstream ofs(filename);
        boost::archive::text_oarchive oa(ofs);
        oa << b;
    }
    // load block
    nl::Block b2("b2", 1, 1, 1);
    {
        std::ifstream ifs(filename);
        boost::archive::text_iarchive ia(ifs);
        ia >> b2; 
    }

    auto dim1 = b.dimensions();
    auto dim2 = b2.dimensions();

    EXPECT_EQ(dim1[0], dim2[0]);
    EXPECT_EQ(dim1[1], dim2[1]);
    EXPECT_EQ(dim1[2], dim2[2]);

    EXPECT_EQ(b.name, b2.name);
    EXPECT_EQ(b.trainable, b2.trainable);
    
    EXPECT_FLOAT_EQ(b.data(0,0,0), b2.data(0,0,0));
    EXPECT_FLOAT_EQ(b.data(0,0,1), b2.data(0,0,1));
    EXPECT_FLOAT_EQ(b.grad(0,0,0), b2.grad(0,0,0));
    EXPECT_FLOAT_EQ(b.grad(0,0,1), b2.grad(0,0,1));

}

#endif // NEURAL_LIB_SERIALIZATION_TEST_H
