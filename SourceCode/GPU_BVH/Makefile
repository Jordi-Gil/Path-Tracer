CUDA_HOME = /usr/local/cuda-10.1
#CUDA_HOME = /Soft/cuda/9.0.176
HOST_COMPILER = g++
NVCC = $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC_DBG = -g -G
NVCC_FLAGS = -I Header/ -std=c++11 
#$(NVCC_DBG)

OBJDIR = Object
1GPUDIR = 1GPU
NGPUDIR = NGPU
HEADERDIR = Header
SOURCEDIR = Source

# Gencode arguments
SMS ?= 61

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
#$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

OBJ = $(addprefix $(OBJDIR)/, $(patsubst Source/%.cu, %.o, $(wildcard Source/*.cu)))

OBJS_1 = $(OBJ) $(OBJDIR)/$(1GPUDIR)/main_1GPU.o
OBJS_2 = $(OBJ) $(OBJDIR)/$(NGPUDIR)/main_NGPUs.o


all: $(OBJDIR) $(OBJDIR)/$(1GPUDIR) $(OBJDIR)/$(NGPUDIR) path_tracing_1GPU path_tracing_NGPUs

$(OBJDIR):
	mkdir $(OBJDIR)
	
$(OBJDIR)/$(1GPUDIR):
	mkdir $(OBJDIR)/$(1GPUDIR)

$(OBJDIR)/$(NGPUDIR):
	mkdir $(OBJDIR)/$(NGPUDIR)

path_tracing_1GPU: $(OBJS_1)
	$(NVCC) -rdc=true $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $(OBJS_1) -lcudadevrt

path_tracing_NGPUs: $(OBJS_2)
	$(NVCC) -rdc=true $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $(OBJS_2) -lcudadevrt

$(OBJDIR)/$(1GPUDIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
	
$(OBJDIR)/$(NGPUDIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
	
$(OBJDIR)/%.o: Source/%.cu Header/%.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

projecte.tar:
	tar -cvf $@ *.cu Source/*.cu Header/*.cuh Makefile *.sh

clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/$(NGPUDIR)/*.o $(OBJDIR)/$(1GPUDIR)/*.o *.png *.tar path_tracing_1GPU path_tracing_NGPU *.e* *.o*
