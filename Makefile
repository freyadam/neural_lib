
SRC=src
BIN=bin
OBJ=obj
EXT=extern

SOURCES=$(wildcard $(SRC)/*.cpp)
HEADERS=$(wildcard $(SRC)/*.hpp)
OBJECTS=$(SOURCES:$(SRC)/%.cpp=$(OBJ)/%.o)
TARGET=$(BIN)/neural_lib

CC=g++
CFLAGS=--std=c++14 -Wall -O2
LDFLAGS=

.PHONY=all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir $(BIN)
	$(CC) $^ $(LDFLAGS) -o $@

$(OBJECTS): $(OBJ)/%.o : $(SRC)/%.cpp
	@mkdir $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	$(foreach object,$(OBJECTS), rm -f $(object)${\n})
	rm -f $(TARGET)
	rm -rf doc

documentation:
	@mkdir doc
	doxygen doxygen_config
