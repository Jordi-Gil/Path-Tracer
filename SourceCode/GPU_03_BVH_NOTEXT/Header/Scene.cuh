#ifndef SCENE_HH
#define SCENE_HH

#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>

#include "Camera.cuh"
#include "Sphere.cuh"
#include "Obj.cuh"
#include "Math.cuh"
#include "Helper.cuh"

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

struct ObjEval{
    
  __host__ __device__ inline bool operator()(Triangle a, Triangle b){
    return (a.getMorton() < b.getMorton());
  }
};

struct ObjEval2{
    
  __host__ __device__ inline bool operator()(Sphere a, Sphere b){
    return (a.getMorton() < b.getMorton());
  }
};



#endif //SCENE_HH
