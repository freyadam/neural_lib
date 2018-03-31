
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

.PHONY: all run clean doc test

all: $(LIB) $(MAIN)

$(LIB): $(LIB_OBJECTS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -shared $^ -o $@

$(MAIN): $(OBJ)/main.o
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@  -Lbin -lneural -lboost_serialization

$(OBJECTS): $(OBJ)/%.o : $(SRC)/%.cpp $(wildcard $(SRC)/*.hpp)
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	LD_LIBRARY_PATH=bin ./$(MAIN)

clean:
	$(foreach object,$(OBJECTS), rm -f $(object)${\n})
	rm -f $(TARGET)
	rm -rf doc

doc:
	@mkdir -p doc
	doxygen doxygen_config

test:	$(LIB)
	make --directory=test all
	LD_LIBRARY_PATH=bin ./test/test_binary
