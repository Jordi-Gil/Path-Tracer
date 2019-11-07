#ifndef _VECTOR3_HH_INCLUDE
#define _VECTOR3_HH_INCLUDE

#include <cmath>
#include <iostream>

class Vector3
{
    
public:
    
    __host__ __device__ Vector3();
    __host__ __device__ Vector3(float x, float y, float z);
    
    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;
    __host__ __device__ float r() const;
    __host__ __device__ float g() const;
    __host__ __device__ float b() const;
    
    __host__ __device__ const Vector3& operator+() const;
    __host__ __device__ Vector3 operator-() const;
    __host__ __device__ float operator[](int i) const;
    __host__ __device__ float& operator[](int i);
    
    __host__ __device__ Vector3& operator+=(const Vector3 &v2);
    __host__ __device__ Vector3& operator-=(const Vector3 &v2);
    __host__ __device__ Vector3& operator*=(const Vector3 &v2);
    __host__ __device__ Vector3& operator/=(const Vector3 &v2);
    
    __host__ __device__ Vector3& operator*=(const float t);
    __host__ __device__ Vector3& operator/=(const float t);
    
    __host__ __device__ static Vector3 Zero();
    __host__ __device__ static Vector3 One();
    
    __host__ __device__ float length() const;
    __host__ __device__ float squared_length() const;
    __host__ __device__ void make_unit_vector();

    float v[3];
};

inline std::istream& operator>>(std::istream &is, Vector3 &t){
    is >> t.v[0] >> t.v[1] >> t.v[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const Vector3 &t){
    os << t.v[0] << " " << t.v[1] << " " << t.v[2];
    return os;
}

__host__ __device__ inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] + v2.v[0], v1.v[1] + v2.v[1], v1.v[2] + v2.v[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] - v2.v[0], v1.v[1] - v2.v[1], v1.v[2] - v2.v[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] * v2.v[0], v1.v[1] * v2.v[1], v1.v[2] * v2.v[2]);
}

__host__ __device__ inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] / v2.v[0], v1.v[1] / v2.v[1], v1.v[2] / v2.v[2]);
}

__host__ __device__ inline Vector3 operator*(float c, const Vector3 &v1){
    return Vector3(c * v1.v[0], c * v1.v[1], c * v1.v[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3 &v1, float c){
    return Vector3(v1.v[0] * c, v1.v[1] * c, v1.v[2] * c);
}

__host__ __device__ inline Vector3 operator/(Vector3 v1, float c){
    return Vector3(v1.v[0] / c, v1.v[1] / c, v1.v[2] / c);
}

__host__ __device__ inline Vector3 unit_vector(Vector3 v){
    return v / v.length();
}

__host__ __device__ inline float dot(const Vector3 &v1, const Vector3 &v2){
    return v1.v[0] * v2.v[0] + v1.v[1] * v2.v[1] + v1.v[2] * v2.v[2];
}

__host__ __device__ inline Vector3 cross(const Vector3 &v1, const Vector3 &v2){
    return Vector3( 
        (v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1]), 
        (-(v1.v[0] * v2.v[2] - v1.v[2] * v2.v[0])),
        (v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0])
    );
}

__host__ __device__ inline Vector3 normalize(const Vector3 &v1) {
  return v1/v1.length();
}

#endif /* _VECTOR3_HH_INCLUDE */
