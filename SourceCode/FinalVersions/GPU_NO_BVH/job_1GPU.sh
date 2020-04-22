#!/bin/bash

for i in 200
do
  
  echo "$i sample(s)"
  
./path_tracing_1GPU -filter 11 15 6 3 3 -f cornell_deer_textures -depth 30 -light ON -sizeX 850 -sizeY 480 -skybox ON -oneTex ON -AAit $i > "time_"$i"_32_cornell_deer_textures.txt" 2> "error.txt"

done
