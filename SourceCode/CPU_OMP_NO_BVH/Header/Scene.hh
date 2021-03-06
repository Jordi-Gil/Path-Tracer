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
#include "Math.hh"
#include "Skybox.hh"

enum loadType {
  FFILE, RANDOM, TRIANGL
};

class Scene {
    
public:
    
  Scene() {}
  Scene(int d, int x, int y) : dist(d), nx(x), ny(y) {}
  
  void loadScene(int loadType, const std::string &filename = "", const bool oneTex = false);
  void sceneRandom();
  void sceneFromFile(const std::string &filename, const bool oneTex);
  void sceneTriangle();
  
  Material loadMaterial(const std::string &line,int type, int texType, bool oneTex = false);
  Obj loadObj(const std::string &line, bool oneTex);
  Sphere loadSphere(const std::string &line);
  int *loadSizes(const std::string &line, int *s);
  mat4 getTransformMatrix(const std::vector<std::string> &transforms, const Vector3 &center);
  Triangle loadTriangle(const std::string &line, int num);
  Camera loadCamera(const std::string &line, int nx, int ny);
  
  Camera getCamera();
  unsigned int getSize();
  Triangle *getObjects();
  Skybox getSkybox();
  unsigned char **getTextures();
  unsigned int getNumTextures();
  Vector3 *getTextureSizes();

private:
  
  Camera cam;
  int dist;
  int nx, ny;
  Triangle *objects;
  Sphere *spheres;
  unsigned int size;
  
  Skybox sky;
  std::vector<unsigned char *> images;
  std::vector<Vector3> textureSizes;
};

#endif //SCENE_HH
