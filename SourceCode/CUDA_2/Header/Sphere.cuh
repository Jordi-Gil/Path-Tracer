#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Material.cuh"

class Sphere {
  
public:
  
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(Vector3 cen, float r, Material mat);
    
  __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  __host__ __device__ Vector3 getCenter();
  __host__ __device__ float getRadius();
  __host__ __device__ Material getMaterial();
  
private:  
  
  Vector3 center;
  float radius;
  Material mat_ptr;
};

#endif /* _SPHERE_HH_INCLUDE */
