#ifndef NEURAL_LIB_NET_H
#define NEURAL_LIB_NET_H

#include <iostream>
#include <unordered_map>
#include <vector>

#include "op.hpp"
#include "block.hpp"
#include "graph.hpp"

namespace nl {

	class Net : public Op {	
    public:
        /// Constructor
        Net(std::string name): Op(name) {}
        /// Insert operation into network.
        /// @param op pointer to operation.
		void add(Op* op);

        void forward();

        void backward();

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
	};

} // namespace nl

#endif // NEURAL_LIB_NET_H
