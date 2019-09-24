#include "Ray.cuh"

__host__ __device__ Ray::Ray() {}

__host__ __device__ Ray::Ray(const Vector3& a, const Vector3& b, float ti) {
    A = a;
    B = b;
    _time = ti;
}

__host__ __device__ Vector3 Ray::origin() const {
    return A;
}

__host__ __device__ Vector3 Ray::direction() const {
    return B;
}

__host__ __device__ float Ray::time() const {
	return _time;
}

__host__ __device__ Vector3 Ray::point_at_parameter(float t) const {
    return A + t*B;
}
