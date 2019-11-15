#ifndef SCENE_HH
#define SCENE_HH

#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>

#include "Camera.hh"
#include "Sphere.hh"
#include "Obj.hh"
#include "Helper.hh"

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::min()

enum loadType {
  FFILE, RANDOM, TRIANGL
};

class Scene {
    
public:
    
  Scene() {}
  Scene(int d, int x, int y) : dist(d), nx(x), ny(y) {}
  
  void loadScene(int loadType, const std::string &filename = "");
  void sceneRandom();
  void sceneFromFile(const std::string &filename);
  void sceneTriangle();
  
  Camera getCamera();
  Sphere *getObjects();
  unsigned int getSize();
  
  Triangle *getTriangles();

private:
  
  Camera cam;
  int dist;
  int nx, ny;
  Sphere *spheres;
  Triangle *triangles;
  unsigned int size;
};

#endif //SCENE_HH
