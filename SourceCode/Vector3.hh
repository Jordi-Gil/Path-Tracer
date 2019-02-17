#ifndef _VECTOR3_HH_INCLUDE
#define _VECTOR3_HH_INCLUDE

#include <cmath>

class Vector3
{
    
public:
    
    Vector3();
    Vector3(float x, float y, float z);
    
    inline float x() const;    
    inline float y() const;
    inline float z() const;
    inline float r() const;
    inline float g() const;
    inline float b() const;
    
    inline const Vector3& operator+() const;
    inline Vector3 operator-() const;
    inline float operator[](int i) const;
    
    inline Vector3& operator+=(const Vector3 &v2);
    inline Vector3& operator-=(const Vector3 &v2);
    inline Vector3& operator*=(const Vector3 &v2);
    inline Vector3& operator/=(const Vector3 &v2);
    inline Vector3& operator*=(const float t);
    inline Vector3& operator/=(const float t);
    
    inline static Vector3 Zero(void);
    inline static Vector3 One(void);
    
    inline float length() const;
    inline float squared_length() const;
    inline void make_unit_vector();

private:
    float v[3];
};

#endif /* _VECTOR3_HH_INCLUDE */
