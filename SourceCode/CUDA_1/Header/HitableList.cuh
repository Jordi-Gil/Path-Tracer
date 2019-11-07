#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Sphere.cuh"

class HitableList {
  
public:
  
    __device__ HitableList();
    __device__ HitableList(Sphere *l, int n);
    __device__ bool checkCollision(const Ray& r, float t_min, float t_max, hit_record& rec);
    __device__ int length();
    __device__ Sphere *getObjects();

private:    
    
    Sphere *list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */
