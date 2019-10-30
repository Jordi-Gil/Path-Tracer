#ifndef SCENE_HH
#define SCENE_HH

#include "Sphere.hh"

class Scene:{
    
public:
    
    Scene() {}
    
    void loadScene();
    
    Sphere *spheres;
    unsigned int numSpheres;
}

#endif //SCENE_HH
