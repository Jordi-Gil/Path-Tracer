#ifndef SCENE_HH
#define SCENE_HH

#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Sphere.hh"
#include "Triangle.hh"
#include "Helper.hh"

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::min()

enum loadType {
  FFILE, RANDOM, TRIANGL
};

class Scene {
    
public:
    
    Scene() {}
    Scene(int d) : dist(d) {}
    
    void loadScene(int loadType, const std::string &filename = "");
    void sceneRandom();
    void sceneFromFile(const std::string &filename);
    void sceneTriangle();
    
    Sphere *getObjects();
    unsigned int getSize();
    
    Triangle *getTriangles();

private:
  
    Sphere *spheres;
    Triangle *triangles;
    unsigned int size;
    int dist;
};

#endif //SCENE_HH
