#ifndef NEURAL_LIB_NET_H
#define NEURAL_LIB_NET_H

#include <iostream>
#include <unordered_map>
#include <vector>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/unordered_map.hpp>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"

namespace nl {

    ///
    /// Class representing networks of operations. Operations are created
    /// outside of the instance of network and then added. Net builds a 
    /// DAG and based on it, it can compute all operations in correct order.
    /// Note that Net is a child of Op so you can recursively add networks 
    /// into other networks.
    ///
	class Net : public Op {	
    public:
        /// Constructor
        Net(std::string name): Op(name) {}
        /// Insert operation into network.
        /// @param op pointer to operation.
		void add(Op* op);

        virtual void forward();

        virtual void backward();

        virtual block_map inputs();

        virtual block_map outputs();

        /// Unordered set of all blocks in the net identified by their names.        
		block_map blocks;
        /// Unordered map of all ops in the net identified by their names.
		unordered_map<std::string, Op*> ops;
        /// Get ordering of operations. Primarily for testing purposes.
        std::vector<Op*> get_ordering() {
            ordering_is_current();
            return ordering;
        }
    private:
        /// Make sure that ordering vector corresponds to current state
        /// of graph.
        void ordering_is_current();
        /// Insert op and its corresponding input and output blocks 
        /// into maps 'ops' and 'blocks'
        void insert_into_maps(Op* op);
        /// True iff some changes were made to the graph (new edge
        /// or vertex was added) after last call of g.get_ordering().
        bool changed = false;
        /// Graph with ordered edges representing relations in network.
        Graph g;
        /// Sequence of operations that describes order of computation
        /// in forward pass. Everything is reversed in backward pass
        std::vector<Op*> ordering;
        
        // Default constructor, for serialization purposes
        Net(): Op("default_name") {}

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<nl::Op>(*this);
            ar & blocks;
            ar & ops;
            ar & g;
            ar & changed;
            ar & ordering;
        }
        friend class boost::serialization::access;
	};

} // namespace nl

#endif // NEURAL_LIB_NET_H
