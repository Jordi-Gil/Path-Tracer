OPTIONS = -O2 -Wall -Wextra -Werror -Wno-uninitialized -Wno-sign-compare -std=c++17

main: main.cc Vector3.hh
	g++ -o main main.cc Vector3.cc

clean:
	rm main

raytracing:
	tar -cvf raytracing.tar main.cc Vector3.cc Vector3.hh Makefile
