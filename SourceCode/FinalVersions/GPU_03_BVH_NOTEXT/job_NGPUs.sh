#!/bin/bash
export PATH=/Soft/cuda/9.0.176/bin:$PATH

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
