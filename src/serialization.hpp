#ifndef NEURAL_LIB_SERIALIZATION_H
#define NEURAL_LIB_SERIALIZATION_H

#include <cassert>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "block.hpp"
#include "neuron.hpp"

/* Note: currently majority of serializing code is in 
   

*/

namespace boost {
    namespace serialization {

    } // namespace serialization
} // namespace boost

BOOST_CLASS_EXPORT_GUID(nl::Op, "Op") 
BOOST_CLASS_EXPORT_GUID(nl::Neuron, "Neuron") 
BOOST_CLASS_EXPORT_GUID(nl::Dense, "Dense")
BOOST_CLASS_EXPORT_GUID(nl::TransferFn, "TransferFn")
BOOST_CLASS_EXPORT_GUID(nl::Sigmoid, "Sigmoid")
BOOST_CLASS_EXPORT_GUID(nl::Tanh, "Tanh")
BOOST_CLASS_EXPORT_GUID(nl::ReLU, "ReLU")
BOOST_CLASS_EXPORT_GUID(nl::Softplus, "Softplus")
BOOST_CLASS_EXPORT_GUID(nl::Linear, "Linear")
BOOST_CLASS_EXPORT_GUID(nl::Net, "Net")

#endif // NEURAL_LIB_SERIALIZATION_H
