OPTIONS = -O2 -Wall -Wextra -Werror -Wno-uninitialized -Wno-sign-compare -std=c++17

main: main.cc Vector3.hh Ray.hh
	g++ -o main main.cc Vector3.cc Ray.cc

clean:
	rm main

raytracing:
	tar -cvf raytracing.tar main.cc Vector3.cc Vector3.hh Makefile
