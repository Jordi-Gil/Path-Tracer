#include "aabb.hh"

aabb::aabb(const Vector3& a, const Vector3& b) {
  
  _min = a; 
  _max = b;
  
}

Vector3 aabb::min() const {
  return _min;
}

Vector3 aabb::max() const {
  return _max;
}
