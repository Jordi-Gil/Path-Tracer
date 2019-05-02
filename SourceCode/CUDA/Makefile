CUDA_HOME 	= /usr/local/cuda
HOST_COMPILER 	= g++
NVCC 		= $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC_DBG	= -g -G
NVCC_FLAGS	= -m64 -std=c++11 -cudart=shared -rdc=true

# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


OBJS = main.o Vector3.o Ray.o Sphere.o HitableList.o Camera.o Material.o

ray_traicing:  $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o ray_traicing $(OBJS)

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

Vector3.o: Vector3.cu Vector3.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Vector3.o -c Vector3.cu

Ray.o: Ray.cu Ray.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Ray.o -c Ray.cu

Sphere.o: Sphere.cu Sphere.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Sphere.o -c Sphere.cu

HitableList.o: HitableList.cu HitableList.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o HitableList.o -c HitableList.cu

Camera.o: Camera.cu Camera.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Camera.o -c Camera.cu

Material.o: Material.cu Material.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Material.o -c Material.cu

pic.ppm: ray_traicing
	rm -f pic.ppm
	./ray_traicing -d > pic.ppm

pic.jpg: pic.ppm
	rm -f pic.jpg
	pnmtojpeg pic.ppm -d > pic.jpg
	xdg-open pic.jpg

profile_basic: ray_traicing
	nvprof ./ray_traicing -d > pic.ppm

profile_metrics: ray_traicing
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./ray_traicing -d > pic.ppm

ray_traicing_cuda.tar:
	tar -cvf $@  *.cu *.cuh Makefile

clean:
	rm -rf ray_traicing *.o pic.ppm pic.jpg *.tar
