# -DOMPI_SKIP_MPICXX:
# this compiler definition is needed to silence warnings caused by the openmpi CXX
# bindings that are deprecated. This is needed on gnu compilers from version 8 forward.
# see: https://github.com/open-mpi/ompi/issues/5157


ifeq ($(macOS), true)
	CXX = clang++
	CXX_FLAGS = -Xpreprocessor -I/usr/local/include -L/usr/local/lib -lomp --std=c++17 -Wall -Wextra -march=native -O3>

	MPICXX = mpicxx
	MPICXX_FLAGS = -Xpreprocessor -I/usr/local/include -L/usr/local/lib -lomp --std=c++17 -Wall -Wextra -march=native  -DOMPI_SKIP_MPICXX -O3>
else
	CXX = c++
	CXX_FLAGS = -O3 -std=c++17 -lm -Wall -Wextra -fopenmp -mavx

	MPICXX = mpicxx
	MPICXX_FLAGS = --std=c++17 -mavx -O3 -Wall -Wextra -g -DOMPI_SKIP_MPICXX
endif

OPENMP = -fopenmp

MODEL_DECLARATION = model/Model.hpp model/Node.hpp
MODEL_COMPILED = model/Model.cpp model/Node.cpp


#-----------------------------------------------------------------------------------------#
all: sequential omp mpi hybrid
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
sequential: gcn_sequential.cpp $(MODEL_DECLARATION) $(MODEL_COMPILED)
	$(CXX) $(CXX_FLAGS) $(OPENMP) -o $@ $< $(MODEL_COMPILED)

run_sequential: 
	./sequential
#-----------------------------------------------------------------------------------------#



#-----------------------------------------------------------------------------------------#
omp: gcn_omp.cpp $(MODEL_DECLARATION) $(MODEL_COMPILED)
	$(CXX) $(CXX_FLAGS) $(OPENMP) -o $@ $< $(MODEL_COMPILED)

run_omp:
	./omp

#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
mpi: gcn_mpi.cpp $(MODEL_DECLARATION) $(MODEL_COMPILED)
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o $@ $< $(MODEL_COMPILED)

run_mpi:
	mpirun -np 6 --use-hwthread-cpus ./mpi

#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
hybrid: gcn_hybrid.cpp $(MODEL_DECLARATION) $(MODEL_COMPILED)
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o $@ $< $(MODEL_COMPILED)

run_hybrid:
	mpirun -np 2 ./hybrid
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
hybrid-simd: gcn_hybrid-simd.cpp $(MODEL_DECLARATION) $(MODEL_COMPILED)
	$(MPICXX) $(MPICXX_FLAGS) $(OPENMP) -o $@ $< $(MODEL_COMPILED)

run_hybrid-simd:
	mpirun -np 2 ./hybrid-simd
#-----------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------#
clean:
	rm -rf *.o sequential omp mpi hybrid
#-----------------------------------------------------------------------------------------#

