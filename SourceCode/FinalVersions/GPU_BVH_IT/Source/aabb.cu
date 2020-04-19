#include "aabb.cuh"

__host__ __device__ aabb::aabb(const Vector3& a, const Vector3& b) {
  
  _min = a; 
  _max = b;
  
}

__host__ __device__ Vector3 aabb::min() const {
  return _min;
}

__host__ __device__ Vector3 aabb::max() const {
  return _max;
}
