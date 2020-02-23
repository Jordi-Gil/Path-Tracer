#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Triangle.cuh"

class HitableList {
  
public:
  
    __device__ HitableList();
    __device__ HitableList(Triangle *l, int n);
    __device__ bool checkCollision(const Ray& r, float t_min, float t_max, hit_record& rec);
    __device__ int length();
    __device__ Triangle *getObjects();

private:    
    
    Triangle *list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */
