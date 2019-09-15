#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Helper.cuh"
#include "Hitable.cuh"
#include "Material.cuh"
#include "aabb.cuh"

class Sphere: public Hitable {
  
public:
    __device__ Sphere() {}
    __device__ Sphere(Vector3 cen, float r, Material *mat);
    
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual void bounding_box(float t0, float t1, aabb &box) const;
    __device__ virtual unsigned int getMorton() const;
    __device__ virtual void setMorton(unsigned int code);
    __device__ virtual aabb getBox() const;
    __device__ virtual Vector3 getCenter() const;
    
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
    unsigned int morton_code;
    aabb box;
};

#endif /* _SPHERE_HH_INCLUDE */
