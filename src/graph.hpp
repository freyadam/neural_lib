#ifndef NEURAL_LIB_GRAPH_H
#define NEURAL_LIB_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>

#include "exceptions.hpp"

using namespace std;

namespace nl {

    /// Oriented graph
    class Graph {
    private:

        /// enum used during topological sort
        enum Mark {
            NONE,    /// vertex was not marked
            CURRENT, /// current search passed through this vertex
            VISITED  /// ended search passed through this vertex
        };

        /// Structure representing a single vertex in the graph
        struct Vertex {
            /// constructor
            /// @param name of the new vertex
            Vertex(string name): name(name), mark(Mark::NONE) {}

            string name; /// unique name of vertex
            vector<Vertex*> outgoing; /// outgoing edges
            vector<Vertex*> incoming; /// incoming edges
            Mark mark; /// used during topogical sort
        };

        ///
        /// Auxiliary function used during topogical sorting. Modified
        /// Depth First search.
        /// @param v vertex from which search is being run
        /// @param vect vector in which results are being stored
        ///
        void visit(Vertex & v, vector<Vertex *> & vect);

        /// Vector of all vertices in a graph
        unordered_map<string, Vertex> vertices;

    public:
    
        /// Add vertex
        /// @param name name of the new vertex
        bool add_vertex(string name);

        /// Add edge
        bool add_edge(string from, string to);

        ///
        /// Get topological order of vertices. If there is no such order, 
        /// throw an exception.
        /// @return Names of vertices in topological order. First vertex 
        /// in position zero, second vertex in position one etc.
        ///    
        vector<string> get_ordering();

    };

} // namespace nl

#endif // NEURAL_LIB_GRAPH_H
