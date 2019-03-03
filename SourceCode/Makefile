OPTIONS = -O2 -Wall -Wextra -Werror -Wno-uninitialized -Wno-sign-compare -std=c++17

main: main.cc Vector3.hh Ray.hh Sphere.hh Hitable.hh HitableList.hh Camera.hh Material.hh
	g++ -o main main.cc Vector3.cc Ray.cc Sphere.cc HitableList.cc Camera.cc Material.cc

clean:
	rm main

raytracing:
	tar -cvf raytracing.tar main.cc Vector3.cc Vector3.hh Makefile
