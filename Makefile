NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
ARCH = -arch=sm_86

NVFLAGS = $(ARCH) -O3 --use_fast_math -Xcompiler "-O2 -Wall -std=c++11 -fopenmp" -std=c++11
CXXFLAGS = -O2 -Wall -std=c++11
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lcufft -lgomp -lm -lstdc++

SRCDIR = src
BINDIR = bin
TARGET = $(BINDIR)/bem_cuda

# Source files
CU_SRCS = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
          $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
          $(SRCDIR)/block_gmres.cu $(SRCDIR)/precond.cu \
          $(SRCDIR)/pfft.cu $(SRCDIR)/surface_pfft.cu
CPP_SRCS = $(SRCDIR)/mesh.cpp $(SRCDIR)/rwg.cpp $(SRCDIR)/rhs.cpp \
           $(SRCDIR)/orient.cpp $(SRCDIR)/output.cpp \
           $(SRCDIR)/go_field.cpp \
           $(SRCDIR)/main.cpp

# Object files
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# CUDA sources
$(SRCDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -c -o $@ $<

# C++ sources
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -x cu -c -o $@ $<

clean:
	rm -f $(SRCDIR)/*.o $(TARGET)

.PHONY: all clean
