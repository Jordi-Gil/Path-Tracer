OPTIONS = -O2 -Wall -Wextra -Werror -Wno-uninitialized -Wno-sign-compare -std=c++17
OPTIONS2 = -O2 -Wall -Wextra -Wno-sign-compare -std=c++17

main: main.cc Vector3.hh Ray.hh Hitable.hh Sphere.hh HitableList.hh Camera.hh Material.hh
	g++ -o main main.cc Vector3.cc Ray.cc Sphere.cc HitableList.cc Camera.cc Material.cc $(OPTIONS2)
