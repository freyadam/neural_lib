#ifndef NEURAL_LIB_EXCEPTIONS_H
#define NEURAL_LIB_EXCEPTIONS_H

namespace nl {

    ///
    /// Exception thrown in the case no topological ordering was possible
    /// on an oriented graph. Generally that means there is a cycle present.
    /// 
    struct TopologicalException : public std::exception {
        const char * what() const throw () {
            return "Topological order could not be created.";
        }
    };
    
    ///
    /// Exception thrown when inserted object would replace some already
    /// present object.
    ///
    struct DuplicityException : public std::exception {
        const char * what() const throw () {
            return "Object with the same name is already present.";
        }
    };

    /// 
    /// Dimensions were not compatible. This may for example happen
    /// when Op is accepting only input of certain size and 
    /// provided block does not fit these parameters.
    ///
    struct DimensionException : public std::exception {
        const char * what() const throw () {
            return "Dimension mismatch.";
        }
    };

    /// 
    /// Unknown option was selected. This may for example happen
    /// if you pick a transfer function that is not implemented.    
    ///
    struct UnknownOptionException : public std::exception {
        const char * what() const throw () {
            return "Given option is not on the list of possible choices.";
        }
    };

    /// 
    /// Input failure. Most likely because input was malformed in some way.
    ///
    struct InputException : public std::exception {
        const char * what() const throw () {
            return "Input failure.";
        }
    };

}

#endif // NEURAL_LIB_EXCEPTIONS_H
