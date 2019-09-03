#ifndef _MOVING_SPHERE_HH_INCLUDE
#define _MOVING_SPHERE_HH_INCLUDE

#include "Hitable.cuh"
#include "Material.cuh"

class MovingSphere: public Hitable {
  
public:
    __device__ MovingSphere() {}
    __device__ MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat): center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(mat){};
    
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    __host__ __device__ int length() const { return -1;}
    
    __device__ Vector3 center(float time) const;
    
    Vector3 center0, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
