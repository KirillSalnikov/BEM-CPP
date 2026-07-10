CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
ARCH ?= -arch=sm_70
OPENMP ?= 1
LAPACK ?= 0
CUSOLVER ?= 0
HOST_OPT ?= -O2
SOLVER_HOST_OPT ?= -O3 -march=native
NVCC_EXTRA_FLAGS ?=
CUDA_RPATH ?= 1

CUDA_TARGET ?= $(CUDA_HOME)/targets/x86_64-linux
CUDA_LIB_DIRS = $(CUDA_TARGET)/lib $(CUDA_HOME)/lib/x86_64-linux-gnu $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib

NVFLAGS = $(ARCH) $(NVCC_EXTRA_FLAGS) -O3 -I$(CUDA_TARGET)/include -Xcompiler "$(HOST_OPT) -Wall -Wno-unknown-pragmas -std=c++11" -std=c++11
CXXFLAGS = $(HOST_OPT) -Wall -std=c++11 -I$(CUDA_TARGET)/include
LDFLAGS = $(addprefix -L,$(CUDA_LIB_DIRS)) -lcudart -lcufft -lm -lstdc++
HOST_TEST_DIR = tests
HOST_CHECKS = \
	$(HOST_TEST_DIR)/operator_config_check \
	$(HOST_TEST_DIR)/precond_policy_check \
	$(HOST_TEST_DIR)/solver_policy_check \
	$(HOST_TEST_DIR)/mesh_quality_check \
	$(HOST_TEST_DIR)/output_json_mesh_check

ifeq ($(CUDA_RPATH),1)
LDFLAGS += $(foreach dir,$(CUDA_LIB_DIRS),-Xlinker -rpath -Xlinker $(dir))
endif

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
          $(SRCDIR)/block_gmres.cu $(SRCDIR)/device_linalg.cu $(SRCDIR)/precond.cu \
          $(SRCDIR)/pfft.cu $(SRCDIR)/surface_pfft.cu
CU_SRCS_FMM = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
              $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
              $(SRCDIR)/block_gmres.cu $(SRCDIR)/device_linalg.cu $(SRCDIR)/precond.cu
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

all: cuda-toolchain-check $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

fmm-only: cuda-toolchain-check $(TARGET_FMM)

cuda-toolchain-check:
	CUDA_HOME="$(CUDA_HOME)" CXX="$(CXX)" python3 scripts/detect_cuda_toolchain.py --summary --require-local-build

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

$(SRCDIR)/solver.o $(SRCDIR)/solver.fmm.o \
$(SRCDIR)/block_gmres.o $(SRCDIR)/block_gmres.fmm.o: HOST_OPT=$(SOLVER_HOST_OPT)

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) $(TARGET_FMM) $(HOST_CHECKS)

$(HOST_TEST_DIR)/operator_config_check: $(HOST_TEST_DIR)/operator_config_check.cpp $(SRCDIR)/*.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/precond_policy_check: $(HOST_TEST_DIR)/precond_policy_check.cpp $(SRCDIR)/precond_policy.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/solver_policy_check: $(HOST_TEST_DIR)/solver_policy_check.cpp $(SRCDIR)/solver_policy.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/mesh_quality_check: $(HOST_TEST_DIR)/mesh_quality_check.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/mesh_quality_check.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/output_json_mesh_check: $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp $(SRCDIR)/output.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp

host-checks: $(HOST_CHECKS)
	$(HOST_TEST_DIR)/operator_config_check
	$(HOST_TEST_DIR)/precond_policy_check
	$(HOST_TEST_DIR)/solver_policy_check
	$(HOST_TEST_DIR)/mesh_quality_check
	$(HOST_TEST_DIR)/output_json_mesh_check
	python3 scripts/check_result_metadata.py --strict /tmp/bem_output_json_mesh_check.json

host-audits: host-checks
	scripts/run_local_audits.sh

audit-1-6: host-audits
	python3 scripts/audit_1_6.py --out runs/audit_1_6_report.json
	python3 scripts/check_audit_1_6_report.py runs/audit_1_6_report.json

audit-1-6-summary:
	python3 scripts/summarize_audit_1_6.py runs/audit_1_6_report.json

cuda-runtime-check:
	@mkdir -p runs/audit_1_6_cuda
	python3 scripts/detect_cuda_toolchain.py --json-out runs/audit_1_6_cuda/cuda_runtime_detect.json --require-runtime

cuda-audits:
	scripts/run_cuda_reference_audits.sh

cuda-audits-summary:
	python3 scripts/summarize_audit_1_6.py runs/audit_1_6_cuda/report.json --require-cuda-reference

.PHONY: all fmm-only cuda-toolchain-check host-checks host-audits audit-1-6 audit-1-6-summary cuda-runtime-check cuda-audits cuda-audits-summary clean
