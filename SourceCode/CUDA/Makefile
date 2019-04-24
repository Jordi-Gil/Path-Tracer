CUDA_HOME 	= /usr/local/cuda
HOST_COMPILER 	= g++
NVCC 		= $(CUDA_HOME)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC_DBG	= -g -G
NVCC_FLAGS	= $(NVCC_DBG) -m64 -std=c++11 -cudart=shared -rdc=true

GENCODE_FLAGS	= -gencode arch=compute_60,code=sm_60

OBJS = main.o Vector3.o Ray.o 

ray_traicing:  $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o ray_traicing $(OBJS)

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o main.o -c main.cu

Vector3.o: Vector3.cu Vector3.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Vector3.o -c Vector3.cu

Ray.o: Ray.cu Ray.cuh
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o Ray.o -c Ray.cu

pic.ppm: ray_traicing
	rm -f pic.ppm
	./ray_traicing > pic.ppm

pic.jpg: pic.ppm
	rm -f pic.jpg
	pnmtojpeg pic.ppm > pic.jpg
	xdg-open pic.jpg

profile_basic: ray_traicing
	nvprof ./ray_traicing > pic.ppm

profile_metrics: ray_traicing
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./ray_traicing > pic.ppm

clean:
	rm -rf ray_traicing *.o pic.ppm pic.jpg
