#ifndef _MOVING_SPHERE_HH_INCLUDE
#define _MOVING_SPHERE_HH_INCLUDE

#include "Helper.cuh"
#include "Hitable.cuh"
#include "Material.cuh"
#include "aabb.cuh"

class MovingSphere: public Hitable {
  
public:
    __host__ __device__ MovingSphere() {}
    __host__ __device__ MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat);
    __host__ __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
	__host__ __device__ virtual void bounding_box(float t0, float t1, aabb& box) const;
    __host__ __device__ virtual unsigned int getMorton() const;
    __host__ __device__ virtual void setMorton(unsigned int code);
    __host__ __device__ virtual aabb getBox() const;
    __host__ __device__ virtual Vector3 getCenter() const;
    __host__ __device__ Vector3 get_center(float time) const;
    
    Vector3 center, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
    long long morton_code;
    aabb box;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
