#include "aabb.cuh"

__device__ aabb::aabb(const Vector3& a, const Vector3& b) {
  
  _min = a; 
  _max = b;
  
}

__device__ Vector3 aabb::min() const {
  return _min;
}

__device__ Vector3 aabb::max() const {
  return _max;
}
