OPTIONS = -O2 -Wall -Wextra -Werror -Wno-uninitialized -Wno-sign-compare -std=c++17
OPTIONS2 = -O2 -Wall -Wextra -Wno-sign-compare -std=c++17 -Werror=unknown-pragmas

OMP = -fopenmp

OBJECTS = main.o Vector3.o Ray.o Sphere.o HitableList.o Camera.o Material.o MovingSphere.o

main.exe: main.cc $(OBJECTS) Hitable.hh 
	g++ -o main $(OBJECTS) $(OMP)

main.o: main.cc
	g++ -c main.cc $(OPTIONS) $(OMP) -lpthread

Vector3.o: Vector3.cc Vector3.hh
	g++ -c Vector3.cc $(OPTIONS)

Ray.o: Ray.cc Ray.hh
	g++ -c Ray.cc $(OPTIONS)

Sphere.o: Sphere.cc Sphere.hh
	g++ -c Sphere.cc $(OPTIONS)

HitableList.o: HitableList.cc HitableList.hh
	g++ -c HitableList.cc $(OPTIONS)

Camera.o: Camera.cc Camera.hh
	g++ -c Camera.cc $(OPTIONS)

Material.o: Material.cc Material.hh
	g++ -c Material.cc $(OPTIONS2)

MovingSphere.o: MovingSphere.cc MovingSphere.hh
	g++ -c MovingSphere.cc $(OPTIONS)

clean:
	rm -rf *.o main
