#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Material.cuh"
#include "aabb.cuh"

class Sphere {
  
public:
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(Vector3 cen, float r, Material mat);
  __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  __host__ __device__ void bounding_box(aabb& box);
  __host__ __device__ long long getMorton();
  __host__ __device__ void setMorton(long long code);
  __host__ __device__ aabb getBox();
  __host__ __device__ Vector3 getCenter();
    
  Vector3 center;
  float radius;
  Material mat_ptr;
  long long morton_code;
  aabb box;
};

struct ObjEval{
    
  __host__ __device__ inline bool operator()(Sphere a, Sphere b){
    return (a.getMorton() < b.getMorton());
  }

};

#endif /* _SPHERE_HH_INCLUDE */
