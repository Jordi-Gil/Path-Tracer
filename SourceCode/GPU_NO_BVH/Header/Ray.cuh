#ifndef _RAY_HH_INCLUDE
#define _RAY_HH_INCLUDE

#include "Vector3.cuh"

class Ray
{
  
public:
  __host__ __device__  Ray();
  __host__ __device__  Ray(const Vector3& a, const Vector3& b, float ti = 0.0);
  __host__ __device__  Vector3 origin() const;
  __host__ __device__  Vector3 direction() const;
  __host__ __device__  float time() const;
  __host__ __device__  Vector3 point_at_parameter(float t) const;
  
private:
  
  Vector3 A;
  Vector3 B;
  float _time;
};

#endif /* _RAY_HH_INCLUDE */
