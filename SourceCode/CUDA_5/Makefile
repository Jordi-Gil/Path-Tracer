CUDA_HOME = /usr/local/cuda-10.1
HOST_COMPILER = g++
NVCC = $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC_DBG = -g -G
NVCC_FLAGS = -m64 -rdc=true -lcudadevrt -I Header/

OBJDIR = Object
1GPUDIR = 1GPU
NGPUDIR = NGPU
NGPUDYNDIR = NGPUDYN
HEADERDIR = Header
SOURCEDIR = Source

# Gencode arguments
SMS ?= 70

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
OBJS_2 = $(OBJ) $(OBJDIR)/$(NGPUDIR)/mainNGPUs.o
OBJS_3 = $(OBJ) $(OBJDIR)/$(NGPUDYNDIR)/mainNGPUs_dynamic.o


all: $(OBJDIR) $(OBJDIR)/$(1GPUDIR) $(OBJDIR)/$(NGPUDIR) $(OBJDIR)/$(NGPUDYNDIR) path_tracing_1GPU #path_tracing_NGPUs path_tracing_NGPUs_dynamic

$(OBJDIR):
	mkdir $(OBJDIR)
	
$(OBJDIR)/$(1GPUDIR):
	mkdir $(OBJDIR)/$(1GPUDIR)

$(OBJDIR)/$(NGPUDIR):
	mkdir $(OBJDIR)/$(NGPUDIR)

$(OBJDIR)/$(NGPUDYNDIR):
	mkdir $(OBJDIR)/$(NGPUDYNDIR)

path_tracing_1GPU: $(OBJS_1)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $(OBJS_1)

path_tracing_NGPUs: $(OBJS_2)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $(OBJS_2)

path_tracing_NGPUs_dynamic: $(OBJS_3)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $(OBJS_3)
	
$(OBJDIR)/$(1GPUDIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
$(OBJDIR)/$(NGPUDIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<
	
$(OBJDIR)/$(NGPUDYNDIR)/%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJDIR)/%.o: Source/%.cu Header/%.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

projecte.tar:
	tar -cvf $@ *.cu Source/*.cu Header/*.cuh Makefile *.sh

clean:
	rm -rf $(OBJDIR)/*.o $(OBJDIR)/$(NGPUDYNDIR)/*.o $(OBJDIR)/$(NGPUDIR)/*.o $(OBJDIR)/$(1GPUDIR)/*.o *.ppm *.tar path_tracing_1GPU path_tracing_NGPU path_tracing_NGPUs_dynamic *.e* *.o*
