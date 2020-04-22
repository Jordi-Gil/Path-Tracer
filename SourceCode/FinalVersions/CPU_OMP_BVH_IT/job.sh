for i in 1 10 50
do
  
  echo "$i sample(s)"
  
./path_tracing_sec -filter 11 15 6 3 3 -f CapeHill_cristal_bunny -depth 30 -light ON -sizeX 850 -sizeY 480 -skybox ON -oneTex ON -AAit $i > "time_"$i"_32_CapeHill_cristal_bunny.txt" 2> "error.txt"

done
