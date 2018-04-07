#ifndef NEURAL_LIB_GRAPH_H
#define NEURAL_LIB_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#include "exceptions.hpp"

using namespace std;

namespace nl {

    /// Oriented graph
    class Graph {
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

    private:

        /// enum used during topological sort
        enum Mark {
            NONE,    /// vertex was not marked
            CURRENT, /// current search passed through this vertex
            VISITED  /// ended search passed through this vertex
        };

        /// Structure representing a single vertex in the graph
        class Vertex {
        public:
            /// constructor
            /// @param name of the new vertex
            Vertex(string name): name(name), mark(Mark::NONE) {}

            string name; /// unique name of vertex
            vector<Vertex*> outgoing; /// outgoing edges
            vector<Vertex*> incoming; /// incoming edges
            Mark mark; /// used during topogical sort

            // Default constructor, for serialization purposes
            Vertex(): Vertex("default_name") {}

            // serialization code
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                ar & name;
                ar & mark;
                // to get rid of an "boost::archive::archive_exception"
                // pointer conflict, 'outgoing' and 'incoming' vertices
                // need to be serialized separately
                // For a reasoning, check for example:
                // http://www.bnikolic.co.uk/blog/cpp-boost-ser-conflict.html
            }
            friend class boost::serialization::access;
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
            
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & vertices;

            // get all keys in vertices
            std::vector<std::string> keys;
            for (auto & pair : vertices) 
                keys.push_back(pair.first);

            // sort keys 
            std::sort(keys.begin(), keys.end());

            for (auto & key : keys) {
                ar & vertices[key].incoming;
                ar & vertices[key].outgoing;
            }
            
        }

        friend class boost::serialization::access;
    };

} // namespace nl

#endif // NEURAL_LIB_GRAPH_H


