#ifndef _AABB_HH_INCLUDE
#define _AABB_HH_INCLUDE

#include "Ray.cuh"

__host__  __device__ inline float ffmin(float a, float b) {return a < b ? a : b;}
__host__  __device__ inline float ffmax(float a, float b) {return a > b ? a : b;}

__host__  __device__ inline Vector3 fmin(Vector3 a, Vector3 b) {
  return (a.x() < b.x() && a.y() < b.y() && a.z() < b.z()) ? a : b;
}

__host__  __device__ inline Vector3 fmax(Vector3 a, Vector3 b) {
  return (a.x() > b.x() && a.y() > b.y() && a.z() > b.z()) ? a : b;
}

class aabb {

public:
  
  __host__ __device__ aabb() {}
  __host__ __device__ aabb(const Vector3& a, const Vector3& b);

  __host__ __device__ Vector3 min() const;
  __host__ __device__ Vector3 max() const;

  
  __device__ inline bool hit(const Ray& r, float tmin, float tmax) const {
    
    Vector3 invD = Vector3::One()/r.direction();
    
    Vector3 t0 = (min() - r.origin()) * invD;
    Vector3 t1 = (max() - r.origin()) * invD;
    
    Vector3 tsmaller = fmin(t0, t1);
    Vector3 tbigger = fmax(t0, t1);
    
    tmin = fmax(tmin, fmax(tsmaller[0], fmax(tsmaller[1], tsmaller[2])));
    tmax = fmin(tmax, fmin(tbigger[0] , fmin(tbigger[1], tbigger[2])));
    
    return (tmin < tmax);
    
  }

  
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
