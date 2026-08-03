CUDA_HOME ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda,/usr)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
VERSION := $(strip $(shell cat VERSION 2>/dev/null || printf unknown))
ARCH ?= -arch=sm_70
OPENMP ?= 1
LAPACK ?= 0
CUSOLVER ?= 0
HOST_OPT ?= -O2
SOLVER_HOST_OPT ?= -O3 -march=native
NVCC_EXTRA_FLAGS ?=
CUDA_RPATH ?= 1
FP32_STREAM_FLAGS ?= --default-stream per-thread
FP32_PRECISION_FLAGS ?= -DBEM_DEFAULT_FMM_NEAR_FP32 -DBEM_PFFT_FP32 \
	-DBEM_FMM_CONCURRENT_MEDIA_DEFAULT -DBEM_MULLER_GPU_ASSEMBLY_DEFAULT

CUDA_TARGET ?= $(if $(wildcard $(CUDA_HOME)/targets/x86_64-linux/include),$(CUDA_HOME)/targets/x86_64-linux,$(CUDA_HOME))
CUDA_LIB_DIRS = $(CUDA_TARGET)/lib $(CUDA_HOME)/lib/x86_64-linux-gnu $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib
HOST_CUDA_INCLUDE := $(if $(wildcard $(CUDA_TARGET)/include/cuda_runtime.h),-I$(CUDA_TARGET)/include,-Itests/host_cuda_stubs)

VERSION_FLAGS = -DBEM_VERSION=\"$(VERSION)\"
NVFLAGS = $(ARCH) -ccbin $(CXX) $(NVCC_EXTRA_FLAGS) $(VERSION_FLAGS) -O3 -I$(CUDA_TARGET)/include -Xcompiler "$(HOST_OPT) -Wall -Wno-unknown-pragmas -std=c++11" -std=c++11
CXXFLAGS = $(HOST_OPT) -Wall -std=c++11 $(VERSION_FLAGS) $(HOST_CUDA_INCLUDE)
LDFLAGS = $(addprefix -L,$(CUDA_LIB_DIRS)) -lcudart -lcufft -lcusparse -lm -lstdc++
HOST_TEST_DIR = tests
CUDA_HESSIAN_CHECK = $(HOST_TEST_DIR)/fmm_hessian_check
CUDA_PFFT_HESSIAN_CHECK = $(HOST_TEST_DIR)/pfft_hessian_check
CUDA_MULLER_FMM_CHECK = $(HOST_TEST_DIR)/muller_fmm_check
HOST_CHECKS = \
	$(HOST_TEST_DIR)/operator_config_check \
	$(HOST_TEST_DIR)/precond_policy_check \
	$(HOST_TEST_DIR)/solver_policy_check \
	$(HOST_TEST_DIR)/mesh_quality_check \
	$(HOST_TEST_DIR)/muller_nodal_check \
	$(HOST_TEST_DIR)/muller_dense_check \
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
MULLER_DEMO = $(BINDIR)/muller_nodal_demo
MULLER_FMM_DEMO = $(BINDIR)/muller_nodal_fmm_demo
MULLER_FMM_FP32_DEMO = $(BINDIR)/muller_nodal_fmm_demo_fp32
MULLER_TRAINING_DUMP = $(BINDIR)/muller_training_dump
FP32_BUILD_DIR = build/muller-fp32

# Source files
CU_SRCS = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
          $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
          $(SRCDIR)/block_gmres.cu $(SRCDIR)/device_linalg.cu $(SRCDIR)/precond.cu \
          $(SRCDIR)/pfft.cu $(SRCDIR)/surface_pfft.cu \
          $(SRCDIR)/muller_fmm_gpu.cu
CU_SRCS_FMM = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
              $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
              $(SRCDIR)/block_gmres.cu $(SRCDIR)/device_linalg.cu $(SRCDIR)/precond.cu \
              $(SRCDIR)/muller_fmm_gpu.cu
CPP_SRCS = $(SRCDIR)/mesh.cpp $(SRCDIR)/rwg.cpp $(SRCDIR)/rhs.cpp \
           $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp \
           $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp \
           $(SRCDIR)/muller_fmm.cpp $(SRCDIR)/muller_mbj_fmm.cpp \
           $(SRCDIR)/orient.cpp $(SRCDIR)/output.cpp \
           $(SRCDIR)/main.cpp

# Object files
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)
CU_OBJS_FMM = $(CU_SRCS_FMM:.cu=.fmm.o)
CPP_OBJS_FMM = $(CPP_SRCS:.cpp=.fmm.o)
OBJS_FMM = $(CU_OBJS_FMM) $(CPP_OBJS_FMM)
MULLER_FP32_OBJS = \
	$(FP32_BUILD_DIR)/muller_fmm.o \
	$(FP32_BUILD_DIR)/muller_dense.o \
	$(FP32_BUILD_DIR)/muller_mbj.o \
	$(FP32_BUILD_DIR)/muller_mbj_fmm.o \
	$(FP32_BUILD_DIR)/muller_nodal.o \
	$(FP32_BUILD_DIR)/muller_duffy.o \
	$(FP32_BUILD_DIR)/mesh.o \
	$(FP32_BUILD_DIR)/orient.o \
	$(FP32_BUILD_DIR)/muller_fmm_gpu.o \
	$(FP32_BUILD_DIR)/muller_mbj_gpu.o \
	$(FP32_BUILD_DIR)/muller_paired_gmres.o \
	$(FP32_BUILD_DIR)/fmm.o \
	$(FP32_BUILD_DIR)/p2p.o \
	$(FP32_BUILD_DIR)/pfft.o

$(CU_OBJS) $(CPP_OBJS) $(CU_OBJS_FMM) $(CPP_OBJS_FMM) $(MULLER_FP32_OBJS): Makefile

all: cuda-toolchain-check $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

$(MULLER_DEMO): tools/muller_nodal_demo.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_OPT) -Wall -std=c++11 -I$(SRCDIR) -o $@ $^
	@echo "Built: $@"

$(MULLER_FMM_DEMO): tools/muller_nodal_fmm_demo.cpp \
		$(SRCDIR)/muller_fmm.o $(SRCDIR)/muller_dense.o \
		$(SRCDIR)/muller_mbj.o $(SRCDIR)/muller_mbj_fmm.o \
		$(SRCDIR)/muller_nodal.o \
		$(SRCDIR)/muller_duffy.o $(SRCDIR)/mesh.o \
		$(SRCDIR)/orient.o \
		$(SRCDIR)/muller_fmm_gpu.o $(SRCDIR)/fmm.o \
		$(SRCDIR)/muller_mbj_gpu.o \
		$(SRCDIR)/muller_paired_gmres.o \
		$(SRCDIR)/p2p.o $(SRCDIR)/pfft.o
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas
	@echo "Built: $@"

$(MULLER_FMM_FP32_DEMO): HOST_OPT=-O3 -march=native
$(MULLER_FMM_FP32_DEMO): ARCH=-arch=sm_86
$(MULLER_FMM_FP32_DEMO): tools/muller_nodal_fmm_demo.cpp \
		$(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) \
		-I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas
	@echo "Built optimized mixed-precision solver: $@"

muller-fp32: cuda-toolchain-check $(MULLER_FMM_FP32_DEMO)

$(MULLER_TRAINING_DUMP): tools/muller_training_dump.cpp \
		$(SRCDIR)/muller_fmm.o $(SRCDIR)/muller_nodal.o \
		$(SRCDIR)/muller_duffy.o $(SRCDIR)/mesh.o \
		$(SRCDIR)/muller_fmm_gpu.o $(SRCDIR)/fmm.o \
		$(SRCDIR)/p2p.o $(SRCDIR)/pfft.o
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)
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

$(FP32_BUILD_DIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/*.h
	@mkdir -p $(FP32_BUILD_DIR)
	$(NVCC) $(NVFLAGS) $(FP32_STREAM_FLAGS) \
		$(FP32_PRECISION_FLAGS) -c -o $@ $<

$(FP32_BUILD_DIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	@mkdir -p $(FP32_BUILD_DIR)
	$(NVCC) $(NVFLAGS) $(FP32_STREAM_FLAGS) \
		$(FP32_PRECISION_FLAGS) \
		-x cu -c -o $@ $<

$(SRCDIR)/%.fmm.o: $(SRCDIR)/%.cu $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -DBEM_FMM_ONLY -c -o $@ $<

$(SRCDIR)/%.fmm.o: $(SRCDIR)/%.cpp $(SRCDIR)/*.h
	$(NVCC) $(NVFLAGS) -DBEM_FMM_ONLY -x cu -c -o $@ $<

$(SRCDIR)/solver.o $(SRCDIR)/solver.fmm.o \
$(SRCDIR)/block_gmres.o $(SRCDIR)/block_gmres.fmm.o: HOST_OPT=$(SOLVER_HOST_OPT)

clean:
	rm -f $(SRCDIR)/*.o $(TARGET) $(TARGET_FMM) $(HOST_CHECKS) \
		$(MULLER_DEMO) $(MULLER_FMM_DEMO) $(MULLER_TRAINING_DUMP) \
		$(MULLER_FMM_FP32_DEMO) \
		$(CUDA_HESSIAN_CHECK) $(CUDA_PFFT_HESSIAN_CHECK) \
		$(CUDA_MULLER_FMM_CHECK)
	rm -rf $(FP32_BUILD_DIR)

$(CUDA_HESSIAN_CHECK): $(HOST_TEST_DIR)/fmm_hessian_check.cu $(SRCDIR)/fmm.o $(SRCDIR)/p2p.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

cuda-hessian-check: cuda-toolchain-check $(CUDA_HESSIAN_CHECK)
	$(CUDA_HESSIAN_CHECK)

$(CUDA_PFFT_HESSIAN_CHECK): $(HOST_TEST_DIR)/pfft_hessian_check.cu $(SRCDIR)/pfft.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

cuda-pfft-hessian-check: cuda-toolchain-check $(CUDA_PFFT_HESSIAN_CHECK)
	$(CUDA_PFFT_HESSIAN_CHECK)

$(CUDA_MULLER_FMM_CHECK): $(HOST_TEST_DIR)/muller_fmm_check.cpp \
		$(SRCDIR)/muller_fmm.o $(SRCDIR)/muller_dense.o \
		$(SRCDIR)/muller_mbj.o $(SRCDIR)/muller_mbj_fmm.o \
		$(SRCDIR)/muller_nodal.o $(SRCDIR)/muller_duffy.o \
		$(SRCDIR)/mesh.o $(SRCDIR)/orient.o \
		$(SRCDIR)/muller_fmm_gpu.o \
		$(SRCDIR)/fmm.o $(SRCDIR)/p2p.o \
		$(SRCDIR)/pfft.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

cuda-muller-fmm-check: cuda-toolchain-check $(CUDA_MULLER_FMM_CHECK)
	$(CUDA_MULLER_FMM_CHECK)

cuda-muller-edge-check: cuda-toolchain-check $(CUDA_MULLER_FMM_CHECK)
	$(CUDA_MULLER_FMM_CHECK) --shape prism --ref 0 --ka 1 \
		--ri 1.3 --edge-mode hdiv --max-leaf 512 \
		--digits 5 --near-radius 3

$(HOST_TEST_DIR)/operator_config_check: $(HOST_TEST_DIR)/operator_config_check.cpp $(SRCDIR)/*.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/precond_policy_check: $(HOST_TEST_DIR)/precond_policy_check.cpp $(SRCDIR)/precond_policy.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/solver_policy_check: $(HOST_TEST_DIR)/solver_policy_check.cpp $(SRCDIR)/solver_policy.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/mesh_quality_check: $(HOST_TEST_DIR)/mesh_quality_check.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/mesh_quality_check.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/muller_nodal_check: $(HOST_TEST_DIR)/muller_nodal_check.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/muller_duffy.h $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/muller_nodal_check.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/muller_dense_check: $(HOST_TEST_DIR)/muller_dense_check.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_dense.h $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_mbj.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/muller_duffy.h $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/muller_dense_check.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/output_json_mesh_check: $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp $(SRCDIR)/output.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp

host-checks: $(HOST_CHECKS)
	$(HOST_TEST_DIR)/operator_config_check
	$(HOST_TEST_DIR)/precond_policy_check
	$(HOST_TEST_DIR)/solver_policy_check
	$(HOST_TEST_DIR)/mesh_quality_check
	$(HOST_TEST_DIR)/muller_nodal_check
	$(HOST_TEST_DIR)/muller_dense_check
	$(HOST_TEST_DIR)/output_json_mesh_check
	python3 scripts/check_result_metadata.py --strict /tmp/bem_output_json_mesh_check.json

host-audits: host-checks
	python3 scripts/mueller_audit.py --self-test
	python3 scripts/operator_block_audit.py --self-test
	@for test in tests/test_*.py; do \
		echo "==> $$test"; \
		python3 "$$test" || exit $$?; \
	done

cuda-runtime-check:
	@mkdir -p runs/audit_1_6_cuda
	python3 scripts/detect_cuda_toolchain.py --json-out runs/audit_1_6_cuda/cuda_runtime_detect.json --require-runtime

.PHONY: all fmm-only muller-fp32 cuda-toolchain-check host-checks host-audits cuda-runtime-check cuda-muller-edge-check clean
