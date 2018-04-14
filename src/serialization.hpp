#ifndef NEURAL_LIB_SERIALIZATION_H
#define NEURAL_LIB_SERIALIZATION_H

#include <cassert>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include "block.hpp"
#include "neuron.hpp"

BOOST_CLASS_EXPORT_GUID(nl::Op, "Op") 
BOOST_CLASS_EXPORT_GUID(nl::Neuron, "Neuron") 
BOOST_CLASS_EXPORT_GUID(nl::Dense, "Dense")
BOOST_CLASS_EXPORT_GUID(nl::Conv, "Conv")
BOOST_CLASS_EXPORT_GUID(nl::MaxPool, "MaxPool")
BOOST_CLASS_EXPORT_GUID(nl::CsvReader, "CsvReader")
BOOST_CLASS_EXPORT_GUID(nl::ImgReader, "ImgReader")
BOOST_CLASS_EXPORT_GUID(nl::TransferFn, "TransferFn")
BOOST_CLASS_EXPORT_GUID(nl::Sigmoid, "Sigmoid")
BOOST_CLASS_EXPORT_GUID(nl::Tanh, "Tanh")
BOOST_CLASS_EXPORT_GUID(nl::ReLU, "ReLU")
BOOST_CLASS_EXPORT_GUID(nl::Softplus, "Softplus")
BOOST_CLASS_EXPORT_GUID(nl::Linear, "Linear")
BOOST_CLASS_EXPORT_GUID(nl::Net, "Net")

#endif // NEURAL_LIB_SERIALIZATION_H
