#!/bin/bash
#export PATH=/Soft/cuda/8.0.61/bin:$PATH

### Directivas para el gestor de colas
# Asegurar que el job se ejecuta en el directorio actual
#$ -cwd
# Asegurar que el job mantiene las variables de entorno del shell lamador
#$ -V
# Cambiar el nombre del job
#$ -N PathTracing_1GPU 
# Cambiar el shell
#$ -S /bin/bash


cuda-memcheck ./path_tracing_1GPU -filter 11 15 6 -f cornell_buddha -depth 25 -light ON -sizeX 640 -sizeY 360 -skybox OFF -AAit 50
#./path_tracing_1GPU -filter 11 15 6 -f little -depth 10 -light ON -sizeX 640 -sizeY 360 -skybox ON
