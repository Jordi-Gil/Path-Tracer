#ifndef _RAY_HH_INCLUDE
#define _RAY_HH_INCLUDE

#include "Vector3.cuh"

class Ray
{
public:
    __device__ Ray();
    __device__ Ray(const Vector3& a, const Vector3& b);
    __device__ Vector3 origin() const;
    __device__ Vector3 direction() const;
    
    __device__ Vector3 point_at_parameter(float t) const;
    
    Vector3 A;
    Vector3 B;
};

#endif /* _RAY_HH_INCLUDE */
