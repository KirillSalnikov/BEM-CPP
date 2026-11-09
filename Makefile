CUDA_HOME ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda,/usr)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
MSTM_FC ?= gfortran
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
LDFLAGS = $(addprefix -L,$(CUDA_LIB_DIRS)) -lcudart -lcufft -lcusparse -lm -ldl -lstdc++
HOST_TEST_DIR = tests
CUDA_HESSIAN_CHECK = $(HOST_TEST_DIR)/fmm_hessian_check
CUDA_PFFT_HESSIAN_CHECK = $(HOST_TEST_DIR)/pfft_hessian_check
CUDA_MULLER_FMM_CHECK = $(HOST_TEST_DIR)/muller_fmm_check
HOST_CHECKS = \
	$(HOST_TEST_DIR)/octree_depth_check \
	$(HOST_TEST_DIR)/host_muller_correction_check \
	$(HOST_TEST_DIR)/p2p_radial_factors_check \
	$(HOST_TEST_DIR)/host_cpu_checker_config_check \
	$(HOST_TEST_DIR)/resident_sector_solver_check \
	$(HOST_TEST_DIR)/sector_transform_host_check \
	$(HOST_TEST_DIR)/resident_inverse_layout_check \
	$(HOST_TEST_DIR)/host_orbit_inverse_check \
	$(HOST_TEST_DIR)/ritz_sector_solver_check \
	$(HOST_TEST_DIR)/physical_ritz_prior_check \
	$(HOST_TEST_DIR)/selected_operator_rows_check \
	$(HOST_TEST_DIR)/host_face_transpose_check \
	$(HOST_TEST_DIR)/muller_radiation_actions_check \
	$(HOST_TEST_DIR)/bdm_mesh_transfer_check \
	$(HOST_TEST_DIR)/bdm_mesh_transfer_points_check \
	$(HOST_TEST_DIR)/prism_edge_phase_basis_check \
	$(HOST_TEST_DIR)/prism_sparse_inverse_check \
	$(HOST_TEST_DIR)/planar_muller_symbol_check \
	$(HOST_TEST_DIR)/planar_bdm_symbol_check \
	$(HOST_TEST_DIR)/streaming_host_gmres_check \
	$(HOST_TEST_DIR)/streaming_flexible_check \
	$(HOST_TEST_DIR)/streaming_recycle_check \
	$(HOST_TEST_DIR)/streaming_recycle_merge_check \
	$(HOST_TEST_DIR)/sector_recycle_budget_check \
	$(HOST_TEST_DIR)/sector_recycle_rebudget_check \
	$(HOST_TEST_DIR)/streaming_two_level_check \
	$(HOST_TEST_DIR)/streaming_deferred_strict_check \
	$(HOST_TEST_DIR)/host_hybrid_scheduler_check \
	$(HOST_TEST_DIR)/host_hybrid_budget_check \
	$(HOST_TEST_DIR)/host_hybrid_batch_check \
	$(HOST_TEST_DIR)/muller_cpu_cross_map_check \
	$(HOST_TEST_DIR)/host_fmm_batch_check \
	$(HOST_TEST_DIR)/host_spectral_inverse_check \
	$(HOST_TEST_DIR)/tmatrix_symmetry_warm_start_check \
	$(HOST_TEST_DIR)/tmatrix_rhs_assembly_check \
	$(HOST_TEST_DIR)/sphere_interpolation_check \
	$(HOST_TEST_DIR)/alpha_fourier_check \
	$(HOST_TEST_DIR)/prism_dihedral_check \
	$(HOST_TEST_DIR)/operator_config_check \
	$(HOST_TEST_DIR)/precond_policy_check \
	$(HOST_TEST_DIR)/solver_policy_check \
	$(HOST_TEST_DIR)/mesh_quality_check \
	$(HOST_TEST_DIR)/krylov_deflation_check \
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
          $(SRCDIR)/muller_fmm_gpu.cu $(SRCDIR)/regular_face_fft.cu
CU_SRCS_FMM = $(SRCDIR)/assembly.cu $(SRCDIR)/pmchwt.cu $(SRCDIR)/solver.cu $(SRCDIR)/farfield.cu \
              $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.cu $(SRCDIR)/bem_fmm.cu $(SRCDIR)/gmres.cu \
              $(SRCDIR)/block_gmres.cu $(SRCDIR)/device_linalg.cu $(SRCDIR)/precond.cu \
              $(SRCDIR)/muller_fmm_gpu.cu $(SRCDIR)/regular_face_fft.cu
CPP_SRCS = $(SRCDIR)/mesh.cpp $(SRCDIR)/rwg.cpp $(SRCDIR)/rhs.cpp \
           $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp \
           $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp \
           $(SRCDIR)/muller_fmm.cpp $(SRCDIR)/muller_mbj_fmm.cpp $(SRCDIR)/prism_face_fft.cpp \
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
	$(FP32_BUILD_DIR)/prism_face_fft.o $(FP32_BUILD_DIR)/regular_face_fft.o \
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

# Direct object rebuilds must use the same architecture/optimization as the
# optimized executables. Otherwise a later link silently reuses sm_70/O2 code.
$(MULLER_FP32_OBJS): HOST_OPT=-O3 -march=native
$(MULLER_FP32_OBJS): ARCH=-arch=sm_86

all: cuda-toolchain-check $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

$(MULLER_DEMO): tools/muller_nodal_demo.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) $(HOST_OPT) -Wall -std=c++11 -I$(SRCDIR) -o $@ $^
	@echo "Built: $@"

$(BINDIR)/muller_tmatrix_probe: tools/muller_tmatrix_probe.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) $(HOST_OPT) -Wall -std=c++11 $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) -o $@ $^

$(MULLER_FMM_DEMO): tools/muller_nodal_fmm_demo.cpp $(SRCDIR)/alpha_fourier.h \
		$(SRCDIR)/prism_face_fft.o $(SRCDIR)/regular_face_fft.o \
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
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas
	@echo "Built: $@"

$(MULLER_FMM_FP32_DEMO): HOST_OPT=-O3 -march=native
$(MULLER_FMM_FP32_DEMO): ARCH=-arch=sm_86
$(MULLER_FMM_FP32_DEMO): tools/muller_nodal_fmm_demo.cpp $(SRCDIR)/alpha_fourier.h \
		$(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) \
		-I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas
	@echo "Built optimized mixed-precision solver: $@"

$(BINDIR)/muller_tmatrix_fmm_server: HOST_OPT=-O3 -march=native -fcx-limited-range -pthread
$(BINDIR)/muller_tmatrix_fmm_server: ARCH=-arch=sm_86
TMATRIX_SERVER_HOST_OBJ ?= build/muller_tmatrix_server_host.o
STREAMING_HOST_HEADERS = $(SRCDIR)/streaming_host_gmres.h $(SRCDIR)/streaming_recycle_space.h $(SRCDIR)/streaming_two_level.h $(SRCDIR)/krylov_deflation.h
$(BINDIR)/muller_tmatrix_fmm_server: tools/muller_tmatrix_fmm_server.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/streaming_polynomial_inverse.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/tmatrix_rhs_assembly.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/block_circulant_fft.h $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/prism_circulant_inverse.h $(SRCDIR)/global_coarse_inverse.cu $(SRCDIR)/global_coarse_inverse.h $(SRCDIR)/prism_wave_basis.h
	@mkdir -p $(BINDIR) build
	@mkdir -p $(dir $(TMATRIX_SERVER_HOST_OBJ))
	$(CXX) $(CXXFLAGS) $(FP32_PRECISION_FLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) -c tools/muller_tmatrix_fmm_server.cpp -o $(TMATRIX_SERVER_HOST_OBJ)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(TMATRIX_SERVER_HOST_OBJ) $(filter-out %.h tools/muller_tmatrix_fmm_server.cpp,$^) $(LDFLAGS) -lcublas -ldl

$(BINDIR)/muller_fmm_action_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_fmm_action_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_fmm_action_probe: tools/muller_fmm_action_probe.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/tmatrix_symmetry_warm_start.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_hotspot_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_hotspot_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_hotspot_probe: tools/muller_hotspot_probe.cpp $(MULLER_FP32_OBJS) $(SRCDIR)/physical_ritz_prior.h $(SRCDIR)/host_thread_affinity.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_tmatrix_neural_actions: HOST_OPT=-O3 -march=native -pthread
$(BINDIR)/muller_tmatrix_neural_actions: ARCH=-arch=sm_86
$(BINDIR)/muller_tmatrix_neural_actions: tools/muller_tmatrix_neural_actions.cpp $(MULLER_FP32_OBJS) $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/physical_ritz_prior.h $(SRCDIR)/coarse_image_experiment.h $(SRCDIR)/ritz_sector_solver.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas -ldl

$(BINDIR)/muller_coarse_mesh_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_coarse_mesh_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_coarse_mesh_probe: tools/muller_coarse_mesh_probe.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/bdm_mesh_transfer.h $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/streaming_sector_gmres.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_circulant_inverse.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/prism_circulant_symmetry.h $(SRCDIR)/block_circulant_symmetry.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/block_circulant_fft.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_recycle_fmm_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_recycle_fmm_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_recycle_fmm_probe: tools/muller_recycle_fmm_probe.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_solver_fft_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_solver_fft_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_solver_fft_probe: tools/muller_solver_fft_probe.cpp $(SRCDIR)/streaming_tfqmr.h $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/streaming_polynomial_inverse.h $(SRCDIR)/streaming_idr.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/block_circulant_fft.h $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/prism_circulant_inverse.h $(SRCDIR)/global_coarse_inverse.cu $(SRCDIR)/global_coarse_inverse.h $(SRCDIR)/prism_wave_basis.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(HOST_TEST_DIR)/block_circulant_fft_check: tests/block_circulant_fft_check.cpp $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/block_circulant_fft.h $(SRCDIR)/host_spectral_inverse.h $(SRCDIR)/krylov_deflation.h
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -ldl

$(HOST_TEST_DIR)/mapped_circulant_fft_check: tests/mapped_circulant_fft_check.cpp $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/block_circulant_fft.h $(SRCDIR)/host_spectral_inverse.h $(SRCDIR)/krylov_deflation.h
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -ldl

$(HOST_TEST_DIR)/host_fmm_batch_check: tests/host_fmm_batch_check.cpp $(SRCDIR)/host_fmm_batch.h $(SRCDIR)/host_fmm_band.h $(SRCDIR)/muller_fmm_cpu_avx512.h $(SRCDIR)/host_shared_near_avx512.h
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -lm -ldl

# Optional native CPU check: intentionally separate from portable host-checks.
$(HOST_TEST_DIR)/host_shared_near_check: tests/host_shared_near_check.cpp $(SRCDIR)/host_shared_near_avx512.h $(SRCDIR)/muller_fmm_cpu_avx512.h
	$(CXX) $(CXXFLAGS) -O3 -march=native $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -lm -ldl

$(HOST_TEST_DIR)/prism_circulant_layout_check: tests/prism_circulant_layout_check.cpp $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

.PHONY: cuda-circulant-check
cuda-circulant-check: $(HOST_TEST_DIR)/block_circulant_fft_check $(HOST_TEST_DIR)/mapped_circulant_fft_check $(HOST_TEST_DIR)/prism_circulant_layout_check
	$(HOST_TEST_DIR)/block_circulant_fft_check
	$(HOST_TEST_DIR)/mapped_circulant_fft_check
	$(HOST_TEST_DIR)/prism_circulant_layout_check

$(HOST_TEST_DIR)/prism_circulant_blocks_check: tests/prism_circulant_blocks_check.cpp $(SRCDIR)/prism_circulant_layout.cpp $(MULLER_FP32_OBJS)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas

$(BINDIR)/muller_mbj_cache_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_mbj_cache_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_mbj_cache_probe: tools/muller_mbj_cache_probe.cpp $(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas

$(FP32_BUILD_DIR)/fmm.o $(SRCDIR)/fmm.o $(SRCDIR)/fmm.fmm.o: $(SRCDIR)/fmm_multilevel.cuh $(SRCDIR)/fmm_profile.cuh $(SRCDIR)/fmm_transpose.cuh
$(FP32_BUILD_DIR)/muller_fmm.o $(SRCDIR)/muller_fmm.o: $(SRCDIR)/muller_fmm_transpose.inc

$(BINDIR)/fmm_vector_action_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/fmm_vector_action_probe: ARCH=-arch=sm_86
$(BINDIR)/fmm_vector_action_probe: tools/fmm_vector_action_probe.cpp $(FP32_BUILD_DIR)/fmm.o $(FP32_BUILD_DIR)/p2p.o $(FP32_BUILD_DIR)/muller_nodal.o $(FP32_BUILD_DIR)/mesh.o
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

$(HOST_TEST_DIR)/p2p_shared_check: ARCH=-arch=sm_86
$(HOST_TEST_DIR)/p2p_shared_check: $(HOST_TEST_DIR)/p2p_shared_check.cu $(FP32_BUILD_DIR)/p2p.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

$(HOST_TEST_DIR)/sphere_interpolation_check: $(HOST_TEST_DIR)/sphere_interpolation_check.cpp $(SRCDIR)/sphere_interpolation.h $(SRCDIR)/sphere_quad.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/streaming_host_gmres_check: $(HOST_TEST_DIR)/streaming_host_gmres_check.cpp $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/streaming_flexible_check: $(HOST_TEST_DIR)/streaming_flexible_check.cpp $(STREAMING_HOST_HEADERS) $(SRCDIR)/streaming_polynomial_inverse.h $(SRCDIR)/streaming_idr.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/streaming_recycle_check: $(HOST_TEST_DIR)/streaming_recycle_check.cpp $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/streaming_recycle_merge_check: $(HOST_TEST_DIR)/streaming_recycle_merge_check.cpp $(STREAMING_HOST_HEADERS) $(SRCDIR)/streaming_sector_gmres.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/sector_recycle_budget_check: tests/sector_recycle_budget_check.cpp $(SRCDIR)/sector_recycle_budget.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -Isrc $< -o $@

$(HOST_TEST_DIR)/sector_recycle_rebudget_check: tests/sector_recycle_rebudget_check.cpp $(SRCDIR)/sector_recycle_rebudget.h $(SRCDIR)/sector_recycle_budget.h $(SRCDIR)/streaming_sector_gmres.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -Isrc $< -o $@

$(HOST_TEST_DIR)/streaming_two_level_check: $(HOST_TEST_DIR)/streaming_two_level_check.cpp $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/streaming_deferred_strict_check: $(HOST_TEST_DIR)/streaming_deferred_strict_check.cpp $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/octree_depth_check: $(HOST_TEST_DIR)/octree_depth_check.cpp $(SRCDIR)/octree.h $(SRCDIR)/types.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) $< -o $@

$(BINDIR)/muller_fft_geometry_export: tools/muller_fft_geometry_export.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/muller_nodal.h $(SRCDIR)/quadrature.h
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_recycle_cpu_probe: tools/muller_recycle_cpu_probe.cpp $(STREAMING_HOST_HEADERS) $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_mbs_seed_cpu_probe: tools/muller_mbs_seed_cpu_probe.cpp $(SRCDIR)/maxwell_exterior_projector.cpp $(SRCDIR)/maxwell_exterior_projector.h $(SRCDIR)/mbs_surface_seed.h $(STREAMING_HOST_HEADERS) $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/mbs_surface_seed_check: $(HOST_TEST_DIR)/mbs_surface_seed_check.cpp $(SRCDIR)/mbs_surface_seed.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/maxwell_exterior_projector_check: $(HOST_TEST_DIR)/maxwell_exterior_projector_check.cpp $(SRCDIR)/maxwell_exterior_projector.cpp $(SRCDIR)/maxwell_exterior_projector.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/streaming_image_space_check: $(HOST_TEST_DIR)/streaming_image_space_check.cpp $(SRCDIR)/streaming_image_space.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -std=c++11 -fopenmp -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/muller_radiation_actions_check: tests/muller_radiation_actions_check.cpp $(SRCDIR)/muller_radiation_actions.h $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_radiation_actions_probe: tools/muller_radiation_actions_probe.cpp $(SRCDIR)/muller_radiation_actions.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/bdm_mesh_transfer_check: tests/bdm_mesh_transfer_check.cpp $(SRCDIR)/bdm_mesh_transfer.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/bdm_mesh_transfer_points_check: tests/bdm_mesh_transfer_points_check.cpp $(SRCDIR)/bdm_mesh_transfer.h $(SRCDIR)/quadrature.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/maxwell_exterior_fmm_check: HOST_OPT=-O3 -march=native
$(BINDIR)/maxwell_exterior_fmm_check: ARCH=-arch=sm_86
$(BINDIR)/maxwell_exterior_fmm_check: tools/maxwell_exterior_fmm_check.cpp $(SRCDIR)/maxwell_exterior_projector.cpp $(SRCDIR)/maxwell_exterior_projector.h $(SRCDIR)/maxwell_exterior_fmm.h $(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_mbs_seed_fmm_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_mbs_seed_fmm_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_mbs_seed_fmm_probe: $(SRCDIR)/maxwell_exterior_projector.cpp $(SRCDIR)/maxwell_exterior_projector.h $(SRCDIR)/maxwell_exterior_fmm.h $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/streaming_sector_gmres.h
$(BINDIR)/muller_mbs_seed_fmm_probe: tools/muller_mbs_seed_cpu_probe.cpp $(SRCDIR)/mbs_surface_seed.h $(STREAMING_HOST_HEADERS) $(MULLER_FP32_OBJS) $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/block_circulant_fft.h $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/prism_circulant_inverse.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -DBEM_MBS_FMM -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_mbs_variants_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_mbs_variants_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_mbs_variants_probe: tools/muller_mbs_variants_probe.cpp $(SRCDIR)/streaming_image_space.h $(SRCDIR)/maxwell_exterior_projector.cpp $(SRCDIR)/maxwell_exterior_projector.h $(SRCDIR)/maxwell_exterior_fmm.h $(SRCDIR)/mbs_surface_seed.h $(STREAMING_HOST_HEADERS) $(MULLER_FP32_OBJS) $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/streaming_sector_gmres.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -DBEM_MBS_FMM -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(BINDIR)/muller_edge_correction_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_edge_correction_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_edge_correction_probe: tools/muller_edge_correction_probe.cpp $(SRCDIR)/prism_edge_correction.h $(STREAMING_HOST_HEADERS) $(MULLER_FP32_OBJS) $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/streaming_sector_gmres.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(HOST_TEST_DIR)/prism_edge_correction_check: tests/prism_edge_correction_check.cpp $(SRCDIR)/prism_edge_correction.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/prism_sparse_inverse_check: tests/prism_sparse_inverse_check.cpp $(SRCDIR)/prism_sparse_inverse.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/prism_vertical_inverse.h $(STREAMING_HOST_HEADERS) $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/tmatrix_symmetry_warm_start_check: $(HOST_TEST_DIR)/tmatrix_symmetry_warm_start_check.cpp $(SRCDIR)/tmatrix_symmetry_warm_start.h $(STREAMING_HOST_HEADERS) $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/tmatrix_rhs_assembly_check: $(HOST_TEST_DIR)/tmatrix_rhs_assembly_check.cpp $(SRCDIR)/tmatrix_rhs_assembly.h $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/tmatrix_rhs_cpu_probe: tools/tmatrix_rhs_cpu_probe.cpp $(SRCDIR)/tmatrix_rhs_assembly.h $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/quadrature.h
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/mesh_capacity_probe: tools/mesh_capacity_probe.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -std=c++11 -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

MSTM_SO3_SOURCES = third_party/mstm/so3_recurrences.f90 third_party/mstm/so3_average.f90 tools/tmatrix_so3_mstm.f90
$(BINDIR)/tmatrix_so3_mstm: $(MSTM_SO3_SOURCES)
	mkdir -p $(BINDIR) build/mstm64
	cd build/mstm64 && $(MSTM_FC) -O3 -fopenmp -ffree-line-length-none -cpp -DBEM_SO3_STORAGE_KIND=8 $(addprefix ../../,$^) -o ../../$@

$(BINDIR)/tmatrix_so3_mstm_fp32: $(MSTM_SO3_SOURCES)
	mkdir -p $(BINDIR) build/mstm32
	cd build/mstm32 && $(MSTM_FC) -O3 -fopenmp -ffree-line-length-none -cpp -DBEM_SO3_STORAGE_KIND=4 $(addprefix ../../,$^) -o ../../$@

$(HOST_TEST_DIR)/alpha_fourier_check: $(HOST_TEST_DIR)/alpha_fourier_check.cpp $(SRCDIR)/alpha_fourier.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/prism_dihedral_check: $(HOST_TEST_DIR)/prism_dihedral_check.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

muller-fp32: cuda-toolchain-check $(MULLER_FMM_FP32_DEMO)

$(MULLER_TRAINING_DUMP): tools/muller_training_dump.cpp \
		$(SRCDIR)/prism_face_fft.o $(SRCDIR)/regular_face_fft.o \
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
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)
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
		$(BINDIR)/muller_tmatrix_probe \
		$(BINDIR)/muller_tmatrix_fmm_server \
		$(BINDIR)/muller_recycle_fmm_probe \
		$(BINDIR)/muller_solver_fft_probe \
		$(HOST_TEST_DIR)/block_circulant_fft_check $(HOST_TEST_DIR)/block_circulant_symmetry_check $(HOST_TEST_DIR)/prism_circulant_layout_check $(HOST_TEST_DIR)/prism_circulant_blocks_check \
		$(HOST_TEST_DIR)/global_coarse_inverse_check $(HOST_TEST_DIR)/prism_wave_basis_check $(HOST_TEST_DIR)/prism_face_coarse_check \
		$(BINDIR)/fmm_vector_action_probe $(HOST_TEST_DIR)/p2p_shared_check $(HOST_TEST_DIR)/p2p_strict_real_check $(HOST_TEST_DIR)/p2p_double_single_check \
		$(BINDIR)/regular_face_fft_probe $(BINDIR)/regular_face_fft_fmm_probe \
		$(BINDIR)/tmatrix_so3_mstm $(BINDIR)/tmatrix_so3_mstm_fp32 \
		$(CUDA_HESSIAN_CHECK) $(CUDA_PFFT_HESSIAN_CHECK) \
		$(CUDA_MULLER_FMM_CHECK)
	rm -rf $(FP32_BUILD_DIR)
	rm -rf build/mstm64 build/mstm32

$(CUDA_HESSIAN_CHECK): $(HOST_TEST_DIR)/fmm_hessian_check.cu $(SRCDIR)/fmm.o $(SRCDIR)/p2p.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

cuda-hessian-check: cuda-toolchain-check $(CUDA_HESSIAN_CHECK)
	$(CUDA_HESSIAN_CHECK)

$(HOST_TEST_DIR)/host_muller_correction_check: $(HOST_TEST_DIR)/host_muller_correction_check.cpp $(SRCDIR)/host_muller_correction.h
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/p2p_radial_factors_check: $(HOST_TEST_DIR)/p2p_radial_factors_check.cpp $(SRCDIR)/p2p_radial_factors.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(SRCDIR)/p2p.o $(FP32_BUILD_DIR)/p2p.o: $(SRCDIR)/p2p_double_single.cuh

$(HOST_TEST_DIR)/p2p_strict_real_check: $(HOST_TEST_DIR)/p2p_strict_real_check.cu $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.h $(SRCDIR)/gpu_select.h $(SRCDIR)/p2p_radial_factors.h $(SRCDIR)/p2p_double_single.cuh $(HOST_TEST_DIR)/p2p_test_buffers.h
	$(NVCC) $(NVFLAGS) $(FP32_STREAM_FLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $< $(LDFLAGS)

.PHONY: cuda-p2p-strict-real-check
cuda-p2p-strict-real-check: cuda-toolchain-check $(HOST_TEST_DIR)/p2p_strict_real_check
	$(HOST_TEST_DIR)/p2p_strict_real_check

$(HOST_TEST_DIR)/p2p_double_single_check: $(HOST_TEST_DIR)/p2p_double_single_check.cu $(SRCDIR)/p2p.cu $(SRCDIR)/fmm.h $(SRCDIR)/gpu_select.h $(SRCDIR)/p2p_radial_factors.h $(SRCDIR)/p2p_double_single.cuh $(HOST_TEST_DIR)/p2p_test_buffers.h
	$(NVCC) $(NVFLAGS) $(FP32_STREAM_FLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $< $(LDFLAGS)

.PHONY: cuda-p2p-double-single-check
cuda-p2p-double-single-check: cuda-toolchain-check $(HOST_TEST_DIR)/p2p_double_single_check
	$(HOST_TEST_DIR)/p2p_double_single_check

$(CUDA_PFFT_HESSIAN_CHECK): $(HOST_TEST_DIR)/pfft_hessian_check.cu $(SRCDIR)/pfft.o
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

cuda-pfft-hessian-check: cuda-toolchain-check $(CUDA_PFFT_HESSIAN_CHECK)
	$(CUDA_PFFT_HESSIAN_CHECK)

$(BINDIR)/regular_face_fft_probe: tools/regular_face_fft_probe.cpp $(SRCDIR)/regular_face_fft.cu $(SRCDIR)/regular_face_fft.h $(SRCDIR)/quadrature.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ tools/regular_face_fft_probe.cpp $(SRCDIR)/regular_face_fft.cu $(LDFLAGS)

cuda-regular-face-fft-check: cuda-toolchain-check $(BINDIR)/regular_face_fft_probe
	$(BINDIR)/regular_face_fft_probe --check

$(BINDIR)/regular_face_fft_fmm_probe: tools/regular_face_fft_probe.cpp $(SRCDIR)/regular_face_fft.cu $(SRCDIR)/regular_face_fft.h $(FP32_BUILD_DIR)/fmm.o $(FP32_BUILD_DIR)/p2p.o
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -DBEM_FACE_FMM_COMPARE -I$(SRCDIR) -o $@ tools/regular_face_fft_probe.cpp $(SRCDIR)/regular_face_fft.cu $(FP32_BUILD_DIR)/fmm.o $(FP32_BUILD_DIR)/p2p.o $(LDFLAGS)

$(CUDA_MULLER_FMM_CHECK): $(HOST_TEST_DIR)/muller_fmm_check.cpp \
		$(SRCDIR)/prism_face_fft.o $(SRCDIR)/regular_face_fft.o \
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

$(BINDIR)/muller_face_fft_check: tools/muller_face_fft_check.cpp $(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas

$(BINDIR)/muller_fft_transpose_probe: tools/muller_fft_transpose_probe.cpp $(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas

$(BINDIR)/muller_constrained_fft_probe: tools/muller_constrained_fft_probe.cpp $(MULLER_FP32_OBJS) $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/block_circulant_fft.cu
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS) -lcublas

# Independent CPU FP64 action / GPU overlap experiment; not a production solver.
$(BINDIR)/muller_cpu_overlap_probe: HOST_OPT=-O3 -march=native -fcx-limited-range -pthread
$(BINDIR)/muller_cpu_overlap_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_cpu_overlap_probe: tools/muller_cpu_overlap_probe.cpp $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_fmm_cpu_avx512.h $(MULLER_FP32_OBJS)
	@mkdir -p $(BINDIR) build
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) -c tools/muller_cpu_overlap_probe.cpp -o build/muller_cpu_overlap_host.o
	$(NVCC) $(NVFLAGS) -o $@ build/muller_cpu_overlap_host.o $(MULLER_FP32_OBJS) $(LDFLAGS) -lcublas

$(HOST_TEST_DIR)/muller_cpu_avx512_check: HOST_OPT=-O3 -march=native -fcx-limited-range
$(HOST_TEST_DIR)/muller_cpu_avx512_check: tests/muller_cpu_avx512_check.cpp $(SRCDIR)/muller_fmm_cpu_avx512.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $< -o $@ -lm

$(BINDIR)/muller_tolerance_probe: HOST_OPT=-O3 -march=native
$(BINDIR)/muller_tolerance_probe: ARCH=-arch=sm_86
$(BINDIR)/muller_tolerance_probe: tools/muller_tolerance_probe.cpp $(STREAMING_HOST_HEADERS) $(MULLER_FP32_OBJS) $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/streaming_sector_gmres.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_orbit_inverse.cpp
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_STREAM_FLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

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

$(HOST_TEST_DIR)/krylov_deflation_check: $(HOST_TEST_DIR)/krylov_deflation_check.cpp $(SRCDIR)/krylov_deflation.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/muller_nodal_check: $(HOST_TEST_DIR)/muller_nodal_check.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/muller_duffy.h $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/muller_nodal_check.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/muller_dense_check: $(HOST_TEST_DIR)/muller_dense_check.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_dense.h $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_mbj.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_nodal.h $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/muller_duffy.h $(SRCDIR)/mesh.cpp $(SRCDIR)/mesh.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/muller_dense_check.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp

$(HOST_TEST_DIR)/output_json_mesh_check: $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp $(SRCDIR)/output.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $(HOST_TEST_DIR)/output_json_mesh_check.cpp $(SRCDIR)/output.cpp

host-checks: $(HOST_CHECKS)
	@set -e; for check in $(HOST_CHECKS); do \
		echo "==> $$check"; "$$check"; \
	done
	python3 scripts/check_result_metadata.py --strict /tmp/bem_output_json_mesh_check.json

host-audits: host-checks
	python3 scripts/mueller_audit.py --self-test
	python3 scripts/operator_block_audit.py --self-test
	@for test in tests/test_*.py; do \
		echo "==> $$test"; \
		python3 "$$test" || exit $$?; \
	done

# Small CPU-only independent BDM/Green fixtures for general surface FFT tests.
$(BINDIR)/export_bdm_fft_patterns: tools/export_bdm_fft_patterns.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $^ -o $@

$(BINDIR)/export_regular_surface_fixture: tools/export_regular_surface_fixture.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $^ -o $@

cuda-runtime-check:
	@mkdir -p runs/audit_1_6_cuda
	python3 scripts/detect_cuda_toolchain.py --json-out runs/audit_1_6_cuda/cuda_runtime_detect.json --require-runtime

.PHONY: all fmm-only muller-fp32 cuda-toolchain-check host-checks host-audits cuda-runtime-check cuda-muller-edge-check clean

$(HOST_TEST_DIR)/global_coarse_inverse_check: tests/global_coarse_inverse_check.cpp $(SRCDIR)/global_coarse_inverse.cu $(SRCDIR)/global_coarse_inverse.h $(SRCDIR)/krylov_deflation.h
	$(NVCC) $(NVFLAGS) -I$(SRCDIR) -o $@ $(filter-out %.h,$^) $(LDFLAGS) -lcublas

$(HOST_TEST_DIR)/prism_wave_basis_check: tests/prism_wave_basis_check.cpp $(SRCDIR)/prism_wave_basis.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

$(HOST_TEST_DIR)/prism_edge_phase_basis_check: tests/prism_edge_phase_basis_check.cpp $(SRCDIR)/prism_edge_phase_basis.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -march=native -Wall -std=c++11 -fopenmp -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

.PHONY: cuda-global-coarse-check
cuda-global-coarse-check: $(HOST_TEST_DIR)/global_coarse_inverse_check $(HOST_TEST_DIR)/prism_wave_basis_check
	$(HOST_TEST_DIR)/global_coarse_inverse_check
	$(HOST_TEST_DIR)/prism_wave_basis_check

$(HOST_TEST_DIR)/streaming_tfqmr_check: tests/streaming_tfqmr_check.cpp $(SRCDIR)/streaming_tfqmr.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -std=c++11 -fopenmp -I$(SRCDIR) -o $@ $<

$(HOST_TEST_DIR)/streaming_qmr_check: tests/streaming_qmr_check.cpp $(SRCDIR)/streaming_qmr.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -std=c++11 -fopenmp -I$(SRCDIR) -o $@ $<

$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/streaming_qmr.h

$(HOST_TEST_DIR)/mbj_adjoint_check: tests/mbj_adjoint_check.cpp $(SRCDIR)/muller_mbj.cpp $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/muller_mbj.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O2 -std=c++11 -fopenmp -I$(SRCDIR) -o $@ $(filter-out %.h,$^)

$(HOST_TEST_DIR)/fmm_transpose_check: ARCH=-arch=sm_86
$(HOST_TEST_DIR)/fmm_transpose_check: tests/fmm_transpose_check.cpp $(FP32_BUILD_DIR)/fmm.o $(FP32_BUILD_DIR)/p2p.o
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) -o $@ $^ $(LDFLAGS)

.PHONY: cuda-adjoint-check
cuda-adjoint-check: $(HOST_TEST_DIR)/streaming_qmr_check $(HOST_TEST_DIR)/mbj_adjoint_check $(HOST_TEST_DIR)/block_circulant_fft_check $(HOST_TEST_DIR)/fmm_transpose_check
	OMP_NUM_THREADS=2 $(HOST_TEST_DIR)/streaming_qmr_check
	OMP_NUM_THREADS=2 $(HOST_TEST_DIR)/mbj_adjoint_check
	$(HOST_TEST_DIR)/block_circulant_fft_check
	$(HOST_TEST_DIR)/fmm_transpose_check

$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/tmatrix_symmetry_warm_start.h

$(HOST_TEST_DIR)/prism_symmetry_sectors_check: tests/prism_symmetry_sectors_check.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/streaming_sector_gmres.h
$(HOST_TEST_DIR)/streaming_sector_gmres_check: tests/streaming_sector_gmres_check.cpp $(SRCDIR)/streaming_sector_gmres.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -fopenmp -std=c++11 -I$(SRCDIR) $< -o $@

$(BINDIR)/prism_packet_rhs: tools/prism_packet_rhs.cpp $(SRCDIR)/tmatrix_rhs_assembly.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/prism_symmetric_circulant.h
$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/prism_vertical_inverse.h
.PHONY: host-symmetry-check
host-symmetry-check: $(HOST_TEST_DIR)/prism_symmetry_sectors_check $(HOST_TEST_DIR)/streaming_sector_gmres_check
	OMP_NUM_THREADS=4 $(HOST_TEST_DIR)/prism_symmetry_sectors_check
	OMP_NUM_THREADS=4 $(HOST_TEST_DIR)/streaming_sector_gmres_check
	OMP_NUM_THREADS=4 $(HOST_TEST_DIR)/block_circulant_symmetry_check
	OMP_NUM_THREADS=4 $(HOST_TEST_DIR)/prism_circulant_layout_check
	OMP_NUM_THREADS=4 $(HOST_TEST_DIR)/prism_face_coarse_check

$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/streaming_sector_gmres.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/resident_sector_operator.cu $(SRCDIR)/resident_sector_operator.h $(SRCDIR)/sector_transform_gpu.cu $(SRCDIR)/sector_transform_gpu.h $(SRCDIR)/host_sector_transform_layout.h $(SRCDIR)/resident_prism_inverse.cu $(SRCDIR)/resident_prism_inverse.h $(SRCDIR)/host_symmetric_inverse_layout.h $(SRCDIR)/orbit_inverse_gpu.cu $(SRCDIR)/orbit_inverse_gpu.h $(SRCDIR)/host_orbit_inverse_layout.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/sector_recycle_budget.h $(SRCDIR)/sector_recycle_rebudget.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/planar_preconditioner_config.h $(SRCDIR)/planar_prism_kernels.h $(SRCDIR)/planar_inverse_kernel.h $(SRCDIR)/planar_bdm_symbol.h $(SRCDIR)/planar_muller_symbol.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/physical_ritz_prior.h $(SRCDIR)/coarse_image_experiment.h $(SRCDIR)/streaming_image_space.h $(SRCDIR)/ritz_sector_solver.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/physical_ritz_images.h $(SRCDIR)/prism_d6h_model_action.h $(SRCDIR)/host_memory_estimate.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/external_fft_action.h

$(HOST_TEST_DIR)/physical_ritz_images_check: tests/physical_ritz_images_check.cpp $(SRCDIR)/physical_ritz_images.h $(SRCDIR)/prism_d6h_model_action.h $(SRCDIR)/host_memory_estimate.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/prepare_edge_phase_prior: tools/prepare_edge_phase_prior.cpp $(SRCDIR)/physical_ritz_prior.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/ritz_sector_solver_check: tests/ritz_sector_solver_check.cpp $(SRCDIR)/ritz_sector_solver.h $(SRCDIR)/coarse_image_experiment.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/physical_ritz_prior_check: tests/physical_ritz_prior_check.cpp $(SRCDIR)/physical_ritz_prior.h
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) -O3 -Wall -std=c++11 -I$(SRCDIR) $< -o $@
$(HOST_TEST_DIR)/prism_symmetry_sectors_check: $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/streaming_sector_gmres.h $(STREAMING_HOST_HEADERS)
$(HOST_TEST_DIR)/prism_symmetry_sectors_check: $(SRCDIR)/prism_vertical_inverse.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/prism_vertical_inverse.h
$(BINDIR)/muller_tmatrix_fmm_server: $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_fmm_cpu_avx512.h $(SRCDIR)/host_hybrid_scheduler.h $(SRCDIR)/host_thread_affinity.h $(SRCDIR)/host_spectral_inverse.h

CPU_FACE_HEADERS = $(SRCDIR)/host_fmm_band.h $(SRCDIR)/host_fmm_batch.h $(SRCDIR)/host_shared_near_avx512.h $(SRCDIR)/host_prism_face_fft.h $(SRCDIR)/host_regular_face_fft.h $(SRCDIR)/host_spectral_inverse.h $(SRCDIR)/prism_face_fft.h $(SRCDIR)/host_memory_estimate.h $(SRCDIR)/host_cpu_checker_config.h
$(BINDIR)/muller_tmatrix_fmm_server $(BINDIR)/muller_cpu_overlap_probe: $(CPU_FACE_HEADERS)
$(HOST_TEST_DIR)/host_fmm_batch_check $(HOST_TEST_DIR)/host_regular_face_fft_check: $(SRCDIR)/host_memory_estimate.h
$(FP32_BUILD_DIR)/prism_face_fft.o: $(SRCDIR)/prism_face_fft.h

$(HOST_TEST_DIR)/host_cpu_checker_config_check: tests/host_cpu_checker_config_check.cpp $(SRCDIR)/host_cpu_checker_config.h $(SRCDIR)/host_memory_estimate.h
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) -O2 -Wall -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/muller_cpu_cross_map_check: tests/muller_cpu_cross_map_check.cpp $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_fmm_cpu_avx512.h $(CPU_FACE_HEADERS)
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -lm -ldl

$(HOST_TEST_DIR)/host_regular_face_fft_check: tests/host_regular_face_fft_check.cpp $(SRCDIR)/host_regular_face_fft.h $(SRCDIR)/host_spectral_inverse.h
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -ldl

$(HOST_TEST_DIR)/host_face_transpose_check: tests/host_face_transpose_check.cpp $(SRCDIR)/host_regular_face_fft.h $(SRCDIR)/host_spectral_inverse.h
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -ldl

$(HOST_TEST_DIR)/selected_operator_rows_check: tests/selected_operator_rows_check.cpp $(SRCDIR)/selected_operator_rows.h
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/host_face_taylor_check: tests/host_face_taylor_check.cpp $(SRCDIR)/host_regular_face_fft.h $(SRCDIR)/host_spectral_inverse.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $< -o $@ -ldl

$(HOST_TEST_DIR)/host_prism_face_taylor_check: tests/host_prism_face_taylor_check.cpp $(SRCDIR)/host_prism_face_taylor.h $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/helmholtz_taylor.h $(CPU_FACE_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $< -o $@ -ldl

.PHONY: host-taylor-checks
host-taylor-checks: $(HOST_TEST_DIR)/host_regular_face_fft_check $(HOST_TEST_DIR)/host_face_taylor_check $(HOST_TEST_DIR)/host_prism_face_taylor_check $(HOST_TEST_DIR)/muller_taylor_kernel_check $(HOST_TEST_DIR)/helmholtz_taylor_check $(HOST_TEST_DIR)/host_fmm_taylor_check $(HOST_TEST_DIR)/muller_taylor_fields_check
	$(HOST_TEST_DIR)/host_regular_face_fft_check
	$(HOST_TEST_DIR)/host_face_taylor_check
	$(HOST_TEST_DIR)/host_prism_face_taylor_check
	$(HOST_TEST_DIR)/muller_taylor_kernel_check
	$(HOST_TEST_DIR)/helmholtz_taylor_check
	$(HOST_TEST_DIR)/host_fmm_taylor_check
	$(HOST_TEST_DIR)/muller_taylor_fields_check

$(HOST_TEST_DIR)/muller_taylor_fields_check: tests/muller_taylor_fields_check.cpp $(SRCDIR)/muller_taylor_fields.h $(SRCDIR)/matrix_free_taylor_moments.h $(SRCDIR)/muller_dense.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/muller_taylor_physical_probe: tools/muller_taylor_physical_probe.cpp tests/taylor_cpu_fixture.h $(SRCDIR)/muller_fmm_taylor.h $(SRCDIR)/muller_taylor_fields.h $(SRCDIR)/matrix_free_taylor_moments.h $(SRCDIR)/muller_fmm.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp $(CPU_FACE_HEADERS) $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/muller_taylor_correction.h $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/helmholtz_taylor.h $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/streaming_host_gmres.h $(SRCDIR)/tmatrix_rhs_assembly.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -ffunction-sections -fdata-sections -DBEM_FMM_ONLY -I$(SRCDIR) $(filter-out %.h,$^) -Wl,--gc-sections -o $@ -ldl

$(HOST_TEST_DIR)/host_fmm_taylor_check: tests/host_fmm_taylor_check.cpp $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/helmholtz_taylor.h $(CPU_FACE_HEADERS)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $< -o $@ -ldl

$(HOST_TEST_DIR)/muller_taylor_action_probe: tools/muller_taylor_action_probe.cpp $(SRCDIR)/muller_fmm_taylor.h $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/helmholtz_taylor.h $(SRCDIR)/muller_taylor_correction.h $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/matrix_free_taylor_moments.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp $(CPU_FACE_HEADERS) $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/streaming_host_gmres.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -DBEM_FMM_ONLY -DBEM_TAYLOR_HOST_FIXTURE -I$(SRCDIR) $(filter-out %.h,$^) -o $@ -ldl

$(HOST_TEST_DIR)/muller_taylor_action_probe $(HOST_TEST_DIR)/muller_taylor_physical_probe: $(SRCDIR)/host_prism_face_taylor.h

$(HOST_TEST_DIR)/muller_cpu_physical_face_probe: tools/muller_cpu_physical_face_probe.cpp tests/physical_face_cpu_fixture.h $(CPU_FACE_HEADERS) $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_taylor_correction.h $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp $(SRCDIR)/tmatrix_symmetry_warm_start.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -DBEM_FMM_ONLY -I$(SRCDIR) $(filter-out %.h,$^) -o $@ -ldl

$(HOST_TEST_DIR)/muller_face_taylor_cpu_probe: tools/muller_face_taylor_probe.cpp tests/physical_face_cpu_fixture.h $(CPU_FACE_HEADERS) $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_fmm_taylor.h $(SRCDIR)/host_prism_face_taylor.h $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/helmholtz_taylor.h $(SRCDIR)/muller_taylor_correction.h $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/matrix_free_taylor_moments.h $(SRCDIR)/streaming_host_gmres.h $(SRCDIR)/tmatrix_symmetry_warm_start.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -DBEM_FMM_ONLY -DBEM_TAYLOR_HOST_FIXTURE -I$(SRCDIR) $(filter-out %.h,$^) -o $@ -ldl

$(HOST_TEST_DIR)/muller_face_taylor_cpu_probe: $(SRCDIR)/host_taylor_central_config.h

$(HOST_TEST_DIR)/muller_taylor_physical_cpu_probe: tools/muller_taylor_native_physical_probe.cpp tests/physical_face_cpu_fixture.h $(CPU_FACE_HEADERS) $(SRCDIR)/muller_fmm_cpu.h $(SRCDIR)/muller_fmm_taylor.h $(SRCDIR)/host_prism_face_taylor.h $(SRCDIR)/host_fmm_taylor.h $(SRCDIR)/helmholtz_taylor.h $(SRCDIR)/muller_taylor_correction.h $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/muller_taylor_fields.h $(SRCDIR)/matrix_free_taylor_moments.h $(SRCDIR)/streaming_host_gmres.h $(SRCDIR)/host_taylor_central_config.h $(SRCDIR)/tmatrix_rhs_assembly.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/muller_duffy.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -Wall -std=c++11 -fopenmp -DBEM_FMM_ONLY -DBEM_TAYLOR_HOST_FIXTURE -I$(SRCDIR) $(filter-out %.h,$^) -o $@ -ldl

$(HOST_TEST_DIR)/matrix_free_incremental_check: tests/matrix_free_incremental_check.cpp $(SRCDIR)/matrix_free_taylor_moments.h
	$(CXX) -O3 -Wall -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/helmholtz_taylor_check: tests/helmholtz_taylor_check.cpp $(SRCDIR)/helmholtz_taylor.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/muller_taylor_kernel_check: tests/muller_taylor_kernel_check.cpp $(SRCDIR)/muller_taylor_kernel.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

.PHONY: host-face-fft-checks
.PHONY: host-osrc-checks
host-osrc-checks: $(HOST_TEST_DIR)/planar_muller_symbol_check $(HOST_TEST_DIR)/planar_bdm_symbol_check $(HOST_TEST_DIR)/planar_inverse_kernel_check $(HOST_TEST_DIR)/planar_preconditioner_config_check $(HOST_TEST_DIR)/planar_prism_kernels_check
	$(HOST_TEST_DIR)/planar_muller_symbol_check
	$(HOST_TEST_DIR)/planar_bdm_symbol_check
	$(HOST_TEST_DIR)/planar_inverse_kernel_check
	$(HOST_TEST_DIR)/planar_preconditioner_config_check
	$(HOST_TEST_DIR)/planar_prism_kernels_check

$(HOST_TEST_DIR)/planar_preconditioner_config_check: tests/planar_preconditioner_config_check.cpp $(SRCDIR)/planar_preconditioner_config.h
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) -O3 -Wall -std=c++11 -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/planar_prism_kernels_check: tests/planar_prism_kernels_check.cpp $(SRCDIR)/planar_prism_kernels.h $(SRCDIR)/planar_inverse_kernel.h $(SRCDIR)/planar_bdm_symbol.h $(SRCDIR)/planar_muller_symbol.h $(SRCDIR)/host_spectral_inverse.h $(SRCDIR)/krylov_deflation.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(HOST_TEST_DIR)
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -ldl -o $@

$(HOST_TEST_DIR)/planar_muller_symbol_check: tests/planar_muller_symbol_check.cpp $(SRCDIR)/planar_muller_symbol.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/planar_bdm_symbol_check: tests/planar_bdm_symbol_check.cpp $(SRCDIR)/planar_bdm_symbol.h $(SRCDIR)/planar_muller_symbol.h $(SRCDIR)/krylov_deflation.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/planar_inverse_kernel_check: tests/planar_inverse_kernel_check.cpp $(SRCDIR)/planar_inverse_kernel.h $(SRCDIR)/planar_bdm_symbol.h $(SRCDIR)/planar_muller_symbol.h $(SRCDIR)/host_spectral_inverse.h $(SRCDIR)/krylov_deflation.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -Wall -std=c++11 -fopenmp -I$(SRCDIR) $(filter-out %.h,$^) -ldl -o $@

host-face-fft-checks: $(HOST_TEST_DIR)/host_regular_face_fft_check $(HOST_TEST_DIR)/muller_cpu_cross_map_check $(HOST_TEST_DIR)/host_fmm_batch_check
	$(HOST_TEST_DIR)/host_regular_face_fft_check
	$(HOST_TEST_DIR)/muller_cpu_cross_map_check
	$(HOST_TEST_DIR)/host_fmm_batch_check

$(HOST_TEST_DIR)/host_hybrid_scheduler_check: tests/host_hybrid_scheduler_check.cpp $(SRCDIR)/host_hybrid_scheduler.h $(STREAMING_HOST_HEADERS)
	$(CXX) $(CXXFLAGS) -pthread -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/host_hybrid_budget_check: tests/host_hybrid_budget_check.cpp $(SRCDIR)/host_hybrid_scheduler.h
	$(CXX) $(CXXFLAGS) -pthread -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/host_hybrid_batch_check: tests/host_hybrid_batch_check.cpp $(SRCDIR)/host_hybrid_scheduler.h
	$(CXX) $(CXXFLAGS) -pthread -I$(SRCDIR) $< -o $@

$(HOST_TEST_DIR)/host_spectral_inverse_check: tests/host_spectral_inverse_check.cpp $(SRCDIR)/host_spectral_inverse.h
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $< -o $@ -ldl

$(BINDIR)/muller_tmatrix_fmm_server $(BINDIR)/muller_solver_fft_probe $(HOST_TEST_DIR)/prism_circulant_layout_check: $(SRCDIR)/prism_circulant_symmetry.h $(SRCDIR)/block_circulant_symmetry.h
$(HOST_TEST_DIR)/prism_circulant_layout_check: $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/tmatrix_symmetry_warm_start.h
$(HOST_TEST_DIR)/block_circulant_symmetry_check: tests/block_circulant_symmetry_check.cpp $(SRCDIR)/block_circulant_symmetry.h
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -fopenmp -std=c++11 -I$(SRCDIR) $< -o $@
host-symmetry-check: $(HOST_TEST_DIR)/block_circulant_symmetry_check $(HOST_TEST_DIR)/prism_circulant_layout_check

$(HOST_TEST_DIR)/prism_face_coarse_check: tests/prism_face_coarse_check.cpp $(SRCDIR)/prism_face_coarse.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(HOST_CUDA_INCLUDE) -O3 -march=native -fopenmp -std=c++11 -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/muller_solver_fft_probe: $(SRCDIR)/prism_face_coarse.h
host-symmetry-check: $(HOST_TEST_DIR)/prism_face_coarse_check

$(HOST_TEST_DIR)/resident_inverse_layout_check: tests/resident_inverse_layout_check.cpp $(SRCDIR)/host_symmetric_inverse_layout.h $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/host_orbit_inverse_check: tools/orbit_inverse_gpu_probe.cpp $(SRCDIR)/host_orbit_inverse_layout.h $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/orbit_inverse_gpu_probe: tools/orbit_inverse_gpu_probe.cpp $(SRCDIR)/orbit_inverse_gpu.cu $(SRCDIR)/orbit_inverse_gpu.h $(SRCDIR)/host_orbit_inverse_layout.h $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -DBEM_TEST_ORBIT_GPU=1 -I$(SRCDIR) $(filter-out %.h,$^) -o $@ $(LDFLAGS) -lcublas

$(BINDIR)/muller_resident_inverse_probe: tools/muller_resident_inverse_probe.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/resident_prism_inverse.cu $(SRCDIR)/resident_prism_inverse.h $(SRCDIR)/host_symmetric_inverse_layout.h $(SRCDIR)/host_orbit_inverse_layout.h $(SRCDIR)/orbit_inverse_gpu.cu $(SRCDIR)/orbit_inverse_gpu.h $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_circulant_inverse.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/block_circulant_fft.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) $(filter-out %.h,$^) -o $@ $(LDFLAGS) -lcublas

$(HOST_TEST_DIR)/sector_transform_host_check: tools/sector_transform_gpu_probe.cpp $(SRCDIR)/host_sector_transform_layout.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(HOST_TEST_DIR)/resident_sector_solver_check: tests/resident_sector_solver_check.cpp $(SRCDIR)/prism_sector_solver.h $(SRCDIR)/ritz_sector_solver.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	$(CXX) $(CXXFLAGS) $(if $(filter 1,$(OPENMP)),-fopenmp) -I$(SRCDIR) $(filter-out %.h,$^) -o $@

$(BINDIR)/sector_transform_gpu_probe: tools/sector_transform_gpu_probe.cpp $(SRCDIR)/sector_transform_gpu.cu $(SRCDIR)/sector_transform_gpu.h $(SRCDIR)/host_sector_transform_layout.h $(SRCDIR)/prism_symmetry_sectors.h $(SRCDIR)/muller_nodal.cpp $(SRCDIR)/mesh.cpp
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) -DBEM_TEST_SECTOR_GPU=1 -I$(SRCDIR) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

$(BINDIR)/muller_resident_composite_probe: tools/muller_resident_composite_probe.cpp $(MULLER_FP32_OBJS) $(STREAMING_HOST_HEADERS) $(SRCDIR)/resident_sector_operator.cu $(SRCDIR)/resident_sector_operator.h $(SRCDIR)/sector_transform_gpu.cu $(SRCDIR)/sector_transform_gpu.h $(SRCDIR)/host_sector_transform_layout.h $(SRCDIR)/resident_prism_inverse.cu $(SRCDIR)/resident_prism_inverse.h $(SRCDIR)/host_symmetric_inverse_layout.h $(SRCDIR)/host_orbit_inverse_layout.h $(SRCDIR)/orbit_inverse_gpu.cu $(SRCDIR)/orbit_inverse_gpu.h $(SRCDIR)/prism_orbit_inverse.cpp $(SRCDIR)/prism_orbit_inverse.h $(SRCDIR)/prism_symmetric_circulant.h $(SRCDIR)/prism_circulant_inverse.cpp $(SRCDIR)/prism_circulant_inverse.h $(SRCDIR)/prism_circulant_layout.cpp $(SRCDIR)/prism_circulant_layout.h $(SRCDIR)/block_circulant_fft.cu $(SRCDIR)/block_circulant_fft.h
	@mkdir -p $(BINDIR)
	$(NVCC) $(NVFLAGS) $(FP32_PRECISION_FLAGS) -I$(SRCDIR) $(filter-out %.h,$^) -o $@ $(LDFLAGS) -lcublas
