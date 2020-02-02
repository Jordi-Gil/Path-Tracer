#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N PathTracing_NGPUs
# Cambiar el shell
#$ -S /bin/bash

./path_tracing_NGPUs -filter 11 15 6 -f cornell -depth 10 -light ON -sizeX 640 -sizeY 360 -nGPUs 4
#nvprof --unified-memory-profiling off --print-gpu-summary ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 1 -f i_1GPUs_N
#nvprof --unified-memory-profiling off --print-gpu-trace --metrics sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 1 -f i_1GPUs_Metrics
#
#./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 2 -f i_2GPUs_N
#nvprof --unified-memory-profiling off --print-gpu-summary ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 2 -f i_2GPUs_GPUTrace
#nvprof --unified-memory-profiling off --print-gpu-trace --metrics sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 2 -f i_2GPUs_Metrics
#
#./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 3 -f i_3GPUs_N
#nvprof --unified-memory-profiling off --print-gpu-summary ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 3 -f i_3GPUs_GPUTrace
#nvprof --unified-memory-profiling off --print-gpu-trace --metrics sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 3 -f i_3GPUs_Metrics
#
#./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 4 -f i_4GPUs_N
#nvprof --unified-memory-profiling off --print-gpu-summary ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 4 -f i_4GPUs_GPUTrace
#nvprof --unified-memory-profiling off --print-gpu-trace --metrics sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization ./path_tracing_NGPUs -nthreads 32 -sizeX 512 -sizeY 512 -AAit 50 -nGPUs 4 -f i_4GPUs_Metrics

