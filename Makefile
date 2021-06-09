CXX=c++
CXX_FLAGS= -O3 -std=c++17 -lm -Wall -Wextra -fopenmp -mavx
OPENMP = -fopenmp 

MPICXX = mpicxx
MPICXX_FLAGS = --std=c++17 -mavx -O3 -Wall -Wextra -g -DOMPI_SKIP_MPICXX
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gnu compilers from version 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157

all: sequential omp mpi hybrid


#-----------------------------------------------------------------------------------------#
sequential: gcn_sequential.cpp Model.cpp Model.hpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) -o sequential gcn_sequential.cpp Model.cpp Node.cpp

run_sequential: 
	./sequential
#-----------------------------------------------------------------------------------------#



#-----------------------------------------------------------------------------------------#
omp: gcn_omp.cpp Model.cpp Model.hpp Node.cpp Node.hpp
	$(CXX) $(CXX_FLAGS) $(OPENMP) -o omp gcn_omp.cpp Model.cpp Node.cpp

run_omp:
	./omp

#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
mpi: gcn_mpi.cpp Model.cpp Model.hpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) -o mpi gcn_mpi.cpp Model.cpp Node.cpp

run_mpi:
	mpirun -np 6 ./mpi

#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
hybrid: gcn_hybrid.cpp Model.cpp Model.hpp Node.cpp Node.hpp
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o hybrid gcn_hybrid.cpp Model.cpp Node.cpp

run_hybrid:
	mpirun -np 2 ./hybrid
#-----------------------------------------------------------------------------------------#


clean:
	rm -rf *.o sequential omp mpi hybrid
