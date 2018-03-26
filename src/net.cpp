
#include "net.hpp"

namespace nl {
            
    void Net::add(Op* op) {

        // inserting itself would result in stack overflow 
        if (op == this)
            throw InputException();

        // current ordering is invalid
        changed = true;

        // insert op and its corresponding input and output blocks 
        // into maps 'ops' and 'blocks'
        insert_into_maps(op);

        // insert op into graph
        g.add_vertex(op->name);

        // add oriented edges leading from this op into the graph
        for (auto & block_pair : op->outputs()) {
            for (auto & op_pair : ops) {
                Op* op2 = op_pair.second;
                // there is a block that is simultaneously output of 'op'
                // and input of 'op2'
                if (op2->inputs()[block_pair.first])
                    g.add_edge(op->name, op2->name);
            }
        }

        // add oriented edges leading to this op into the graph
        for (auto & block_pair : op->inputs()) {
            for (auto & op_pair : ops) {
                Op* op2 = op_pair.second;
                // there is a block that is simultaneously output of 'op2'
                // and input of 'op'
                if (op2->outputs()[block_pair.first])
                    g.add_edge(op2->name, op->name);
            }
        }        
        
    }

    void Net::forward() {
        // make sure that we are working with up-to-date op order
        ordering_is_current();
        // sequentially run all (in reverse)
        for (auto & op : ordering) {
            op->forward();
        }
    } 

    void Net::backward() {
        // make sure that we are working with up-to-date op order
        ordering_is_current();
        // sequetially run all (in reverse)
        for (int16_t i = ordering.size() - 1; i >= 0; --i) {
            ordering[i]->backward();            
        }
    } 

    block_map Net::inputs() {
        block_map map;
        // find blocks that do not serve as output of any op in the net
        for (auto & block_pair : blocks) { // try all blocks
            bool present = false;
            for (auto & op_pair : ops) { 
                Op* op = op_pair.second;
                if (op->outputs()[block_pair.first] != nullptr)
                    present = true;
            }

            if (!present)
                map.insert(block_pair);
        }
        return map;
    }

    block_map Net::outputs() {
        block_map map;
        // find blocks that do not serve as input of any op in the net
        for (auto & block_pair : blocks) { // try all blocks
            bool present = false;
            for (auto & op_pair : ops) { 
                Op* op = op_pair.second;
                if (op->inputs()[block_pair.first] != nullptr)
                    present = true;
            }

            if (!present)
                map.insert(block_pair);
        }
        return map;
    }

    void Net::insert_into_maps(Op* op) {        
        // insert op into map of ops
        if (ops[op->name] != nullptr && // value is in map
            ops[op->name] != op)        // but is not the given op
            throw DuplicityException();
        else
            ops[op->name] = op;
                        
        // insert input blocks into map of blocks
        for (auto & pair : op->inputs()) {
            Block* b = pair.second;
            if (blocks[b->name] != nullptr && // value for given key is in map
                blocks[b->name] != b)         // but it is not the given block
                throw DuplicityException();
            else
                blocks[b->name] = b;
        }

        // insert output blocks into map of blocks
        for (auto & pair : op->outputs()) {
            Block* b = pair.second;
            if (blocks[b->name] != nullptr && // value for given key is in map
                blocks[b->name] != b)         // but it is not the given block
                throw DuplicityException();
            else
                blocks[b->name] = b;            
        }        
    }

    void Net::ordering_is_current() {

        // if there is no change in graph from last update or graph is empty,
        // end straight away
        if (!changed)
            return;

        // get ordering in graph or throw exception if there was some issue
        std::vector<std::string> ordering_names = g.get_ordering();
        std::vector<Op*> new_ordering;

        // translate names to block pointers and update 'ordering'
        for (uint16_t i = 0; i < ordering_names.size(); ++i) {
            new_ordering.push_back(ops[ordering_names[i]]);
        }

        ordering = std::move(new_ordering);

        changed = false;
    }

} // namespace nl
