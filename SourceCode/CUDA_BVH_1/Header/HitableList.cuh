#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Hitable.cuh"

class HitableList: public Hitable {
  
public:
    __device__ HitableList();
    __device__ HitableList(Hitable **l, int n);
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb &box) const;
    __host__ __device__ virtual int length() const;
    
    Hitable **list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */