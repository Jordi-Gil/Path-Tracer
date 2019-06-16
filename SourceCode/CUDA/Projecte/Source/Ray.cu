#include "Ray.cuh"

__device__ Ray::Ray(){}

__device__ Ray::Ray(const Vector3& a, const Vector3& b){
    A = a;
    B = b;
}

__device__ Vector3 Ray::origin() const{
    return A;
}

__device__ Vector3 Ray::direction() const{
    return B;
}

__device__ Vector3 Ray::point_at_parameter(float t) const{
    return A + t*B;
}
