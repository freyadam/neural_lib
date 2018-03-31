#ifndef NEURAL_LIB_GRAPH_TEST_H
#define NEURAL_LIB_GRAPH_TEST_H

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "graph.hpp"
#include "exceptions.hpp"

TEST(GraphTest, AddVertex) {

    nl::Graph g;

    // two distinct vertices
    EXPECT_TRUE(g.add_vertex("a"));
    EXPECT_TRUE(g.add_vertex("b"));

    // duplicate vertex
    EXPECT_FALSE(g.add_vertex("b"));

}

TEST(GraphTest, AddEdge) {

    nl::Graph g;

    EXPECT_TRUE(g.add_edge("a","b"));
    // inserting same edge twice should work without issues
    EXPECT_TRUE(g.add_edge("a","b"));
}

TEST(GraphTest, Ordering1) {

    nl::Graph g;
    
    // empty graph should return empty vector
    EXPECT_TRUE(g.get_ordering().empty());
}


TEST(GraphTest, Ordering2) {

    nl::Graph g;    
    g.add_vertex("a");

    std::vector<std::string> vect = g.get_ordering();
    
    // single-node graph returns only a single node
    EXPECT_EQ(vect.size(), 1);
    EXPECT_TRUE(vect[0] == "a");
}

TEST(GraphTest, Ordering3) {

    nl::Graph g;
    g.add_edge("a","b");
    g.add_edge("a","c");
    g.add_edge("b","c");

    std::vector<std::string> vect = g.get_ordering();

    // 3 vertex DAG
    EXPECT_EQ(vect.size(), 3);
    EXPECT_TRUE(vect[0] == "a");
    EXPECT_TRUE(vect[1] == "b");
    EXPECT_TRUE(vect[2] == "c");
}


TEST(GraphTest, Ordering4) {
    nl::Graph g;
    g.add_edge("a","b");
    g.add_edge("b","c");
    g.add_edge("c","a");

    // cycle created from three vertices
    EXPECT_THROW(g.get_ordering(), nl::TopologicalException);
}

#endif // NEURAL_LIB_GRAPH_TEST_H
