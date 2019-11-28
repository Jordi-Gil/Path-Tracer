#ifndef SCENE_HH
#define SCENE_HH

#include "Camera.hh"
#include "Shape.hh"
#include "Obj.hh"
#include "Math.hh"
#include "Helper.hh"

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
  
  unsigned int getSize();
  
  Triangle *getObjects();

private:
  
  Camera cam;
  int dist;
  int nx, ny;
  Triangle *objects;
  Sphere *spheres;
  unsigned int size;
};

#endif //SCENE_HH
