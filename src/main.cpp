
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>

#include "exceptions.hpp"

#include "op.hpp"
#include "block.hpp"

using namespace std;

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
        Vertex(string name): name(name) {}

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
    void visit(Vertex & v, vector<Vertex *> & vect) {
        // return immediately if vertex is already in the vector
        if (v.mark == Mark::VISITED)
            return;

        // graph contains ordered cycle
        if (v.mark == Mark::CURRENT)
            throw nl::DuplicityException();

        v.mark = Mark::CURRENT;

        // visit all successors
        for (auto succ : v.outgoing) {
         visit(*succ, vect);   
        }
        
        v.mark = Mark::VISITED;

        // push vertex to the end of the vector
        vect.push_back(&v);
    }

    /// Vector of all vertices in a graph
    unordered_map<string, Vertex> vertices;

public:
    
    /// Add vertex
    /// @param name name of the new vertex
    void add_vertex(string name) {
        bool success = vertices.insert(make_pair(name, Vertex(name))).second;        
        if (!success)
            throw nl::DuplicityException();
    }

    /// Add edge
    void add_edge(string from, string to) {
        // try to find vertices
        auto origin = vertices.find(from);
        auto destination = vertices.find(to);
        
        // create nodes if they don't exist
        if (origin == vertices.end()) {
            add_vertex(from);
            origin = vertices.find(from);
        }
        if (destination == vertices.end()) {
            add_vertex(to);
            destination = vertices.find(from);
        }
        
        // add edge
        Vertex * in = &(origin->second);
        Vertex * out = &(destination->second);
        // TBD: if not present
        origin->second.outgoing.push_back(out);
        destination->second.incoming.push_back(in);            
    }

    ///
    /// Get topological order of vertices. If there is no such order, 
    /// throw an exception.
    /// @return Names of vertices in topological order. First vertex 
    /// in position zero, second vertex in position one etc.
    ///    
    vector<string> get_ordering() {

        // create empty vector for results
        vector<Vertex *> order;

        // iterate over all vertices in graph
        for (auto vertex_it : vertices) {
            // all vertices must be marked NONE or VISITED when 
            // search is not running
            assert(vertex_it.second.mark != Mark::CURRENT);

            if (vertex_it.second.mark == Mark::NONE)
                visit(vertex_it.second, order);
        }

        reverse(order.begin(), order.end());
        
    }
};


int main(int argc, char *argv[])
{

    nl::Op op;

    std::cout << "I have no mouth but I have to scream." << std::endl;
      
    return 0;
}
