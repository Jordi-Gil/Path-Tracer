#ifndef _HITABLE_HH_INCLUDE
#define _HITABLE_HH_INCLUDE

#include "Ray.cuh"

class Material;

struct hit_record{
    float t;
    Vector3 point;
    Vector3 normal;
    Material *mat_ptr;
};

class Hitable {
  
public:
    
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record &rec) const = 0;
    
};

#endif /* _HITABLE_HH_INCLUDE */
