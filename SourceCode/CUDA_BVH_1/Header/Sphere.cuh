#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Helper.cuh"
#include "Material.cuh"
#include "aabb.cuh"

class Sphere {
  
public:
	
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vector3 cen, float r, Material *mat);
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
	__host__ __device__ void bounding_box(float t0, float t1, aabb &box);
    __host__ __device__ unsigned int getMorton();
    __host__ __device__ void setMorton(unsigned int code);
    __host__ __device__ aabb getBox();
    __host__ __device__ Vector3 getCenter();
    
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
    unsigned int morton_code;
    aabb box;
};

#endif /* _SPHERE_HH_INCLUDE */
