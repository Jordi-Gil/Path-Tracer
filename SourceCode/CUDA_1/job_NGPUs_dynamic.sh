#!/bin/bash
export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N PathTracing_NGPUs_dynamic 
# Cambiar el shell
#$ -S /bin/bash

nvprof --unified-memory-profiling off --print-gpu-summary --device-buffer-size 100 ./path_tracing_NGPUs_dynamic -nthreads 32 -sizeX 512 -sizeY 512 -AAit 1 -nGPUs 4 -f i_4GPUs
#./path_tracing_NGPUs_dynamic -nthreads 32 -sizeX 512 -sizeY 512 -AAit 128 -nGPUs 4 -f i_4GPUs
