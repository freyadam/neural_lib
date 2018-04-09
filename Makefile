
SRC=src
BIN=bin
OBJ=obj
EXT=extern

SOURCES=$(wildcard $(SRC)/*.cpp)
HEADERS=$(wildcard $(SRC)/*.hpp)
OBJECTS=$(SOURCES:$(SRC)/%.cpp=$(OBJ)/%.o)
LIB_OBJECTS=$(filter-out $(OBJ)/example%, $(filter-out $(OBJ)/main.o, $(OBJECTS)))
MAIN=$(BIN)/main
LIB=$(BIN)/libneural.so

EIGEN_PATH=./extern

CC=g++
CFLAGS=--std=c++14 -Wall -O2 -fPIC -I$(EIGEN_PATH)
LDFLAGS=

.PHONY: all run clean doc test run_ex1

# Compile everything
all: $(LIB) $(MAIN) $(BIN)/example1 $(BIN)/example2

# Compile library
$(LIB): $(LIB_OBJECTS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -shared $^ -o $@

# Compile main program (currently empty)
$(MAIN): $(OBJ)/main.o
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@  -Lbin -lneural -lboost_serialization

# Run main program
run: all
	LD_LIBRARY_PATH=bin ./$(MAIN)

# Compile all object files
$(OBJECTS): $(OBJ)/%.o : $(SRC)/%.cpp $(wildcard $(SRC)/*.hpp)
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# Remove all intermediate files
clean:
	$(foreach object,$(OBJECTS), rm -f $(object)${\n})
	rm -f $(TARGET)
	rm -rf doc
	rm test/*_serialization_test.txt

# Create doxygen documentation
doc:
	@mkdir -p doc
	doxygen doxygen_config

# Run tests
test:	$(LIB)
	make --directory=test all
	LD_LIBRARY_PATH=bin ./test/test_binary

# Compile example1
$(BIN)/example1: $(OBJ)/example1.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@  -Lbin -lneural -lboost_serialization

# Run example1
run_ex1: all
	LD_LIBRARY_PATH=bin ./$(BIN)/example1

# Compile example2 
$(BIN)/example2: $(OBJ)/example2.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@  -Lbin -lneural -lboost_serialization

# Run example2
run_ex2: all
	LD_LIBRARY_PATH=bin ./$(BIN)/example2
