#ifndef _RAY_HH_INCLUDE
#define _RAY_HH_INCLUDE

#include "Vector3.hh"

class Ray
{
public:
  Ray();
  Ray(const Vector3& a, const Vector3& b, float ti = 0.0);
  Vector3 origin() const;
  Vector3 direction() const;
  float time() const;
  Vector3 point_at_parameter(float t) const;
    
private:
    
    Vector3 A;
    Vector3 B;
    float _time;
};

#endif /* _RAY_HH_INCLUDE */
