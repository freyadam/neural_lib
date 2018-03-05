#ifndef NEURAL_LIB_OP_H
#define NEURAL_LIB_OP_H

#include "block.h"

namespace nl {

	class Op {
	
		std::string name;
		
	};

	typedef std::shared_ptr<op> op_ptr;

} // namespace nl

#endif // NEURAL_LIB_OP_H
