CXX = g++
CXXFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. -lm

all: libenergy.so grad_w_armijo bfgs_w_classes bfgs_w_varargs

libenergy.so: energy.c energy.cpp energy.hpp
	$(CC) $(CFLAGS) -c -fPIC energy.c -o energy_c.o
	$(CXX) $(CXXFLAGS) -c -fPIC energy.cpp -o energy_cpp.o
	$(CXX) $(CXXFLAGS) -shared -o libenergy.so energy_c.o energy_cpp.o

grad_w_armijo: libenergy.so grad_w_armijo.o 
	$(CXX) $(CXXFLAGS) grad_w_armijo.o -o grad_w_armijo  $(LDFLAGS) -lenergy


bfgs_w_classes: bfgs_w_classes.o
	$(CXX) $(CXXFLAGS) bfgs_w_classes.o -o bfgs_w_classes  $(LDFLAGS)

bfgs_w_varargs: bfgs_w_varargs.o
	$(CC) $(CFFLAGS) bfgs_w_varargs.o -o bfgs_w_varargs  $(LDFLAGS)


clean: FORCE
	@-rm libenergy.so
	@-rm grad_w_armijo
	@-rm grad_w_varargs
	@-rm grad_w_classes

FORCE:
