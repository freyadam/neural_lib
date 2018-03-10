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

}

#endif // NEURAL_LIB_EXCEPTIONS_H
