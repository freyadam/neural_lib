
#include <iostream>

#include "graph.hpp"

void nl::Graph::visit(Vertex & v, vector<Vertex *> & vect) {
    std::cout << v.name << " " << v.mark << std::endl;

    // return immediately if vertex is already in the vector
    if (v.mark == Mark::VISITED)
        return;

    // graph contains ordered cycle
    if (v.mark == Mark::CURRENT)
        throw nl::TopologicalException();

    v.mark = Mark::CURRENT;

    // visit all successors
    for (auto succ : v.outgoing) {
        std::cout << "recurse" << std::endl;
        visit(*succ, vect);   
    }
        
    v.mark = Mark::VISITED;

    // push vertex to the end of the vector
    vect.push_back(&v);
}

bool nl::Graph::add_vertex(string name) {
    bool success = vertices.insert(make_pair(name, Vertex(name))).second;        
    return success; // fail if there was already a vertex of the same name
}

bool nl::Graph::add_edge(string from, string to) {

    // single-vertex cycles are forbidden
    if (from == to)
        return false;

    // try to find vertices
    auto origin = vertices.find(from);
    auto destination = vertices.find(to);
        
    // add vertices if they don't exist
    if (origin == vertices.end()) {
        if (!add_vertex(from))
            return false;
        origin = vertices.find(from);
    }
    if (destination == vertices.end()) {
        if (!add_vertex(to))
            return false;
        destination = vertices.find(to);
    }
                
    Vertex * from_v = &(origin->second);
    Vertex * to_v = &(destination->second);        

    // add edge if not already present
    if (std::find(from_v->outgoing.begin(), from_v->outgoing.end(), to_v) == from_v->outgoing.end()) {
        from_v->outgoing.push_back(to_v);
    }        
    if (std::find(to_v->incoming.begin(), to_v->incoming.end(), from_v) == to_v->incoming.end()) {
        to_v->incoming.push_back(from_v);            
    }

    return true;
}

vector<string> nl::Graph::get_ordering() {

    // create empty vector for results
    vector<Vertex *> order;

    // iterate over all vertices in graph
    for (auto & vertex_it : vertices) {
        // all vertices must be marked NONE or VISITED when 
        // search is not running
        assert(vertex_it.second.mark != Mark::CURRENT);
        std::cout << "new" << std::endl;
        if (vertex_it.second.mark == Mark::NONE)
            visit(vertex_it.second, order);
    }

    reverse(order.begin(), order.end());

    vector<string> ret;
    for (auto ptr : order) {
        ret.push_back(ptr->name);
    }
    return std::move(ret);
}
