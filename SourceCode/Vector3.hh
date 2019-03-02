#ifndef _VECTOR3_HH_INCLUDE
#define _VECTOR3_HH_INCLUDE

#include <cmath>
#include <iostream>

class Vector3
{
    
public:
    
    Vector3();
    Vector3(float x, float y, float z);
    
    float x() const;    
    float y() const;
    float z() const;
    float r() const;
    float g() const;
    float b() const;
    
    const Vector3& operator+() const;
    Vector3 operator-() const;
    float operator[](int i) const;
    float& operator[](int i);
    
    Vector3& operator+=(const Vector3 &v2);
    Vector3& operator-=(const Vector3 &v2);
    Vector3& operator*=(const Vector3 &v2);
    Vector3& operator/=(const Vector3 &v2);
    
    Vector3& operator*=(const float t);
    Vector3& operator/=(const float t);
    
    static Vector3 Zero(void);
    static Vector3 One(void);
    
    float length() const;
    float squared_length() const;
    void make_unit_vector();
    float dot(const Vector3 &v1, const Vector3 &v2);
    Vector3 cross(const Vector3 &v1, const Vector3 &v2);
    

    float v[3];
};

inline std::istream& operator>>(std::istream &is, Vector3 &t){
    is >> t.v[0] >> t.v[1] >> t.v[2];
}

inline std::ostream& operator<<(std::ostream &os, const Vector3 &t){
    os << t.v[0] << " " << t.v[1] << " " << t.v[2];
}

inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] + v2.v[0], v1.v[1] + v2.v[1], v1.v[2] + v2.v[2]);
}

inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] - v2.v[0], v1.v[1] - v2.v[1], v1.v[2] - v2.v[2]);
}

inline Vector3 operator*(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] * v2.v[0], v1.v[1] * v2.v[1], v1.v[2] * v2.v[2]);
}

inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2){
    return Vector3(v1.v[0] / v2.v[0], v1.v[1] / v2.v[1], v1.v[2] / v2.v[2]);
}

inline Vector3 operator*(const Vector3 &v1, float c){
    return Vector3(v1.v[0] * c, v1.v[1] * c, v1.v[2] * c);
}

inline Vector3 operator/(const Vector3 &v1, float c){
    return Vector3(v1.v[0] / c, v1.v[1] / c, v1.v[2] / c);
}

#endif /* _VECTOR3_HH_INCLUDE */
