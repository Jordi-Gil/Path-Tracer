#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Hitable.cuh"
#include "Material.cuh"

class Sphere: public Hitable {
  
public:
    __device__ Sphere() {}
    __device__ Sphere(Vector3 cen, float r, Material *mat): center(cen), radius(r), mat_ptr(mat){};
    
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    __host__ __device__ int length() const { return -1;}
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
};

#endif /* _SPHERE_HH_INCLUDE */
