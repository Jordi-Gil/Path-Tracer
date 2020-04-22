#!/bin/bash
  
./path_tracing_sec -filter 21 10 10 3 3 -i cornell_normal_50

./path_tracing_sec -filter 21 15 10 3 3 -i cornell_normal_10
./path_tracing_sec -filter 21 15 20 3 3 -i cornell_normal_10
./path_tracing_sec -filter 21 15 40 3 3 -i cornell_normal_10


