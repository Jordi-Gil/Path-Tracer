#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N PathTracing_1GPU 
# Cambiar el shell
#$ -S /bin/bash


./path_tracing_1GPU -filter 11 15 6 -f cornell -depth 10 -light ON -sizeX 640 -sizeY 360
#nvprof --unified-memory-profiling off ./path_tracing_1GPU -nthreads 8 -sizeX 512 -sizeY 512 -AAit 50 -f i_1GPU
#nvprof --unified-memory-profiling off --print-gpu-trace --metrics sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization ./path_tracing_1GPU -nthreads 8 -sizeX 512 -sizeY 512 -AAit 50 -f i_1GPU
