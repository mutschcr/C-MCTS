FLAGS=`python3.9-config --ldflags`
INC=`python3.9 -m pybind11 --includes`
CXXFLAGS=-std=c++11

all: rocksample.so

rocksample.so: coord.o node.o rocksample.o simulator.o 
	clang++ -std=c++11 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.9 -m pybind11 --includes` -o $@ pybind.cpp coord.o node.o rocksample.o simulator.o ${FLAGS}

.cpp.o:
	clang++ $(CXXFLAGS) -c $< $(INC)

coord.o: coord.cpp coord.h utils.h
node.o: node.cpp node.h utils.h
rocksample.o: rocksample.cpp rocksample.h utils.h
simulator.o: simulator.cpp simulator.h

clean:
	rm -f *.o *.gch rocksample.so

