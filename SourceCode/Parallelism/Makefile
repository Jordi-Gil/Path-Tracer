OPM = -fopenmp


main: main.cc
	g++ main.cc -o main 

main-omp: main-omp.cc
	g++ main-omp.cc -o main-omp $(OMP)

clean:
	rm -rf main main-omp
