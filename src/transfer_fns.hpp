#ifndef NEURAL_LIB_TRANSFER_FN_H
#define NEURAL_LIB_TRANSFER_FN_H

#include <cmath>
#include <algorithm>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>

#include "exceptions.hpp"

namespace nl {

    /// Abstract parent class for all transfer function classes.
    class TransferFn {
    public:
        /// Result of function for given value.
        virtual float forward(float x) = 0;
        /// Computes derivative in given value.
        virtual float backward(float x) = 0;
    private:
        // vestigial serialization 
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {}
    };    

    /// Softmax transfer function
    class Sigmoid : public TransferFn {
    public:
        virtual float forward(float x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        virtual float backward(float x) {
            float sigm = forward(x);
            return sigm * (1 - sigm);
        }
    private:
        // vestigial serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & boost::serialization::base_object<nl::TransferFn>(*this);
        }
    };

    /// Hyperbolic tangent transfer function.
    class Tanh : public TransferFn {
    public:
        virtual float forward(float x) {
            float ex = std::exp(x);
            float m_ex = std::exp(-x);
            return (ex - m_ex) / (ex + m_ex);
        }

        virtual float backward(float x) {
            return std::pow(2.0 / (std::exp(x) + std::exp(-x)), 2);
        }
    private:
        // vestigial serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & boost::serialization::base_object<nl::TransferFn>(*this);
        }
    };

    /// Rectified Linear Unit transfer function.
    class ReLU : public TransferFn {
    public:
        virtual float forward(float x) {
            return std::max<float>(0, x);
        }

        virtual float backward(float x) {
            return (x > 0) ? 1 : 0;
        }
    private:
        // vestigial serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & boost::serialization::base_object<nl::TransferFn>(*this);
        }
    };

    /// Softplus transfer function.
    class Softplus : public TransferFn {
        virtual float forward(float x) {
            return std::log( 1 + std::exp(x) );
        }

        virtual float backward(float x) {
            return 1.0 / (1.0 + std::exp(-x));
        }
    private:
        // vestigial serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & boost::serialization::base_object<nl::TransferFn>(*this);
        }
    };

    /// Linear transfer function.
    class Linear : public TransferFn {
        virtual float forward(float x) {
            return x;
        }

        virtual float backward(float) {
            return 1.0;
        }
    private:
        // vestigial serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & boost::serialization::base_object<nl::TransferFn>(*this);
        }
    };

    /// Class that holds instances of all implemented transfer function
    /// classes. Static get() function returns correct transfer function
    /// based on its name. There is no reason to create an instance of this
    /// class.
    class TransferFns {
    public:
        ///
        /// Get reference to transfer function based on its name
        /// Valid names:
        /// "sigmoid" ... Sigmoid
        /// "tanh" ... Tanh
        /// "relu" ... ReLU
        /// "softplus" ... Softplus
        /// "linear" ... Linear
        /// @param name name of the transfer functions
        /// 
        static TransferFn* get(std::string name) {
            if (name == "sigmoid")
                return &sigmoid;
            else if (name == "tanh") 
                return &tanh;
            else if (name == "relu")
                return &relu;
            else if (name == "softplus")
                return &softplus;
            else if (name == "linear")
                return &linear;
            else
                throw UnknownOptionException();
        }
    private:
        static Sigmoid sigmoid;
        static Tanh tanh;
        static ReLU relu;
        static Softplus softplus;
        static Linear linear;
    };

} // namespace nl

#endif // NEURAL_LIB_TRANSFER_FN_H


