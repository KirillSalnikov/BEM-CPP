CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
ARCH ?= -arch=sm_70
OPENMP ?= 1
LAPACK ?= 0
CUSOLVER ?= 0
HOST_OPT ?= -O2
SOLVER_HOST_OPT ?= -O3 -march=native

CUDA_TARGET ?= $(CUDA_HOME)/targets/x86_64-linux

NVFLAGS = $(ARCH) -O3 -I$(CUDA_TARGET)/include -Xcompiler "$(HOST_OPT) -Wall -Wno-unknown-pragmas -std=c++11" -std=c++11
CXXFLAGS = $(HOST_OPT) -Wall -std=c++11
LDFLAGS = -L$(CUDA_TARGET)/lib -L$(CUDA_HOME)/lib64 -lcudart -lcufft -lm -lstdc++

ifeq ($(OPENMP),1)
NVFLAGS += -Xcompiler -fopenmp
LDFLAGS += -Xcompiler -fopenmp
endif

ifeq ($(LAPACK),1)
NVFLAGS += -DBEM_USE_LAPACK
LDFLAGS += -llapack -lblas
endif

ifeq ($(CUSOLVER),1)
NVFLAGS += -DBEM_USE_CUSOLVER
LDFLAGS += -lcusolver -lcublas
endif

SRCDIR = src
BINDIR = bin
TARGET = $(BINDIR)/bem_cuda
TARGET_FMM = $(BINDIR)/bem_cuda_fmm

# Source files
CU_SRCS = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
          $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
          $(SRCDIR)/block_gmres.cu $(SRCDIR)/precond.cu \
          $(SRCDIR)/pfft.cu $(SRCDIR)/surface_pfft.cu
CU_SRCS_FMM = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
              $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
              $(SRCDIR)/block_gmres.cu $(SRCDIR)/precond.cu
CPP_SRCS = $(SRCDIR)/mesh.cpp $(SRCDIR)/rwg.cpp $(SRCDIR)/rhs.cpp \
           $(SRCDIR)/orient.cpp $(SRCDIR)/output.cpp \
           $(SRCDIR)/main.cpp

# Object files
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)
CU_OBJS_FMM = $(CU_SRCS_FMM:.cu=.fmm.o)
CPP_OBJS_FMM = $(CPP_SRCS:.cpp=.fmm.o)
OBJS_FMM = $(CU_OBJS_FMM) $(CPP_OBJS_FMM)

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

fmm-only: $(TARGET_FMM)

$(TARGET_FMM): $(OBJS_FMM)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(filter-out -lcufft,$(LDFLAGS))
	@echo "Built: $@"

# CUDA sources
$(SRCDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

# C++ sources
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -x cu -c -o $@ $<

$(SRCDIR)/%.fmm.o: $(SRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -DBEM_FMM_ONLY -c -o $@ $<

$(SRCDIR)/%.fmm.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -DBEM_FMM_ONLY -x cu -c -o $@ $<

$(SRCDIR)/solver.o $(SRCDIR)/solver.fmm.o: HOST_OPT=$(SOLVER_HOST_OPT)

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) $(TARGET_FMM)

.PHONY: all fmm-only clean
