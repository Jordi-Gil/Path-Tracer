#ifndef SCENE_HH
#define SCENE_HH

#include <algorithm>
#include <limits>
#include <string>
#include <fstream>
#include <stdexcept>

#include "Sphere.cuh"

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::min()

enum loadType {
  FFILE, RANDOM
};

class Scene {
    
public:
    
    Scene() {}
    Scene(int d) : dist(d) {}
    
    void loadScene(int loadType, const std::string &filename = "");
    void sceneRandom();
    void sceneFromFile(const std::string &filename);
    
    Sphere *getObjects();
    unsigned int getSize();

private:
  
    Sphere *spheres;
    unsigned int size;
    int dist;
};

#endif //SCENE_HH
