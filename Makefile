
SRC=src
BIN=bin
OBJ=obj
EXT=extern

SOURCES=$(wildcard $(SRC)/*.cpp)
HEADERS=$(wildcard $(SRC)/*.hpp)
OBJECTS=$(SOURCES:$(SRC)/%.cpp=$(OBJ)/%.o)
MAIN=$(BIN)/main
LIB=$(BIN)/libneural.so

CC=g++
CFLAGS=--std=c++14 -Wall -O2 -fPIC
LDFLAGS=

.PHONY: all run clean doc test

all: $(LIB) $(MAIN)

$(LIB): $(filter-out $(OBJ)/main.o, $(OBJECTS))
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) -shared $^ -o $@

$(MAIN): $(OBJ)/main.o
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@  -Lbin -lneural

$(OBJECTS): $(OBJ)/%.o : $(SRC)/%.cpp $(SRC)/%.hpp
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	$(foreach object,$(OBJECTS), rm -f $(object)${\n})
	rm -f $(TARGET)
	rm -rf doc

doc:
	@mkdir -p doc
	doxygen doxygen_config

test:	$(LIB)
	make --directory=test run
