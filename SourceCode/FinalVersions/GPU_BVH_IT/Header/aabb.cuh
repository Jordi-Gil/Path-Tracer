#ifndef _AABB_HH_INCLUDE
#define _AABB_HH_INCLUDE

#include "Ray.cuh"

__host__  __device__ inline float ffmin(float a, float b) {return a <= b ? a : b;}
__host__  __device__ inline float ffmax(float a, float b) {return a >= b ? a : b;}

__host__ __device__ inline void swap(float &a, float &b){float t = a; a = b; b = t;}

class aabb {

public:
  
  __host__ __device__ aabb() {}
  __host__ __device__ aabb(const Vector3& a, const Vector3& b);

  __host__ __device__ Vector3 min() const;
  __host__ __device__ Vector3 max() const;

  
  __device__ inline bool hit(const Ray& r, float tmin, float tmax) const {
  
    for (int i = 0; i < 3; i++) {
    
      float invD = 1.0f / r.direction()[i];
      float t0 = (min()[i] - r.origin()[i]) * invD;
      float t1 = (max()[i] - r.origin()[i]) * invD;
      if(invD < 0.0f) swap(t0,t1);
      tmin = t0 > tmin ? t0 : tmin;
      tmax = t1 < tmax ? t1 : tmax;
      if(tmax <= tmin) return false;
    
    }
    return true;
  }
  
private:
  
  Vector3 _min;
  Vector3 _max;
  
};

__host__ __device__ inline aabb surrounding_box(aabb box0, aabb box1) {
  Vector3 small(  fmin(box0.min().x(), box1.min().x()),
                  fmin(box0.min().y(), box1.min().y()),
                  fmin(box0.min().z(), box1.min().z()));
  Vector3 big  (  fmax(box0.max().x(), box1.max().x()),
                  fmax(box0.max().y(), box1.max().y()),
                  fmax(box0.max().z(), box1.max().z()));
  return aabb(small,big);
}

#endif /* _AABB_HH_INCLUDE */
