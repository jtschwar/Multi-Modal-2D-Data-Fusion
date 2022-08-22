CXX = g++
CXXFLAGS = -fPIC -Wno-div-by-zero -shared -std=c++11 -undefined dynamic_lookup
CXXOPT = -O3 -mcpu=apple-m1 
EIGEN = -I ../thirdparty/eigen
PYBIND11 = `python3 -m pybind11 --includes` 
EDX_CONFIG = ../EDX/ctvlib`python3-config --extension-suffix`

all: ctvlib 

ctvlib: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(EDX_CONFIG)

clean:
	rm -rf ../EDX/*.so ../EDX/*.o 