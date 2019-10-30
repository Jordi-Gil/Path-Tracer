#include "Scene.hh"

void Scene::loaScene(){
    
    numSpheres = 5;
    
    spheres = new Sphere[numSpheres];
    
    
    spheres[0].center = Vector3(0,-1000,0);
    spheres[0].center1 = Vector3(0,-1000,0);
    spheres[0].time0 = 0.0;
    spheres[0].time1 = 1.0;
    spheres[0].radius = 1000;
    spheres[0].mat_ptr = new Lambertian(Vector3(0.5,0.5,0.5));
    
    spheres[1].center = Vector3(0,1,0);
    spheres[1].center1 = Vector3(0,1,0);
    spheres[1].time0 = 0.0;
    spheres[1].time1 = 1.0;
    spheres[1].radius = 1.0;
    spheres[1].mat_ptr = new Dielectric(Vector3::One(),1.5));
    
    spheres[2].center = Vector3(0,1,0);
    spheres[2].center1 = Vector3(0,1,0);
    spheres[2].time0 = 0.0;
    spheres[2].time1 = 1.0;
    spheres[2].radius = 1.0;
    spheres[2].mat_ptr = new Lambertian(Vector3(0.4,0.2,0.1));

    spheres[3].center = Vector3(4,1,0);
    spheres[3].center1 = Vector3(4,1,0);
    spheres[3].time0 = 0.0;
    spheres[3].time1 = 1.0;
    spheres[3].radius = 1.0;
    spheres[3].mat_ptr = new Metal(Vector3(0.7,0.6,0.5),0.0));
    
    spheres[4].center = Vector3(4,1,5);
    spheres[4].center1 = Vector3(4,1,5);
    spheres[4].time0 = 0.0;
    spheres[4].time1 = 1.0;
    spheres[4].radius = 1.0;
    spheres[4].mat_ptr = new Metal(Vector3(0.9, 0.2, 0.2),0.0));
    
}
