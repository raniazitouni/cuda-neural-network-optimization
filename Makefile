NVCC        = nvcc
GCC         = gcc
NVCCFLAGS   = -O2 -Xcompiler -fopenmp
GCCFLAGS    = -O2 -fopenmp -lm

# Binaries
OPT_BIN     = nn_optimized
REF_BIN     = nn_reference

# Sources 
OPT_SRC     = src/optimized_cuda.cu
REF_SRC     = reference_code/nn_cuda.cu

.PHONY: all optimized reference clean help

all: optimized reference

## Build our optimized alternative strategy
optimized: $(OPT_SRC)
	$(NVCC) $(NVCCFLAGS) -o $(OPT_BIN) $(OPT_SRC)
	@echo "Built: $(OPT_BIN)"

## Build the reference baseline
reference: $(REF_SRC)
	$(NVCC) $(NVCCFLAGS) -o $(REF_BIN) $(REF_SRC)
	@echo "Built: $(REF_BIN)"

## Remove compiled binaries
clean:
	rm -f $(OPT_BIN) $(REF_BIN)
	@echo "Cleaned"

## Show help
help:
	@echo ""
	@echo "  make all        — build both optimized and reference binaries"
	@echo "  make optimized  — build our alternative strategy only"
	@echo "  make reference  — build the reference baseline only"
	@echo "  make clean      — remove compiled binaries"
	@echo ""
	@echo "  Then run:  ./run_optimized.sh   or   ./run_compare.sh"
	@echo ""
