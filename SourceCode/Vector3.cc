#include "Vector3.hh"

Vector3::Vector3(){
    
}
    
Vector3::Vector3(float x, float y, float z){
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

inline float Vector3::x() const {
    return v[0];
}

inline float Vector3::y() const {
    return v[1];
}

inline float Vector3::z() const {
    return v[2];
}

inline float Vector3::r() const {
    return v[0];
}

inline float Vector3::g() const {
    return v[1];
}

inline float Vector3::b() const {
    return v[2];
}

inline const Vector3& Vector3::operator+() const {
    return *this;
}

inline Vector3 Vector3::operator-() const {
    return Vector3(-v[0],-v[1],-v[2]);
}

inline float Vector3::operator[](int i) const {
    return v[i];
}

inline Vector3 Vector3::Zero(void){
    return Vector3(0.f,0.f,0.f);
}

inline Vector3 Vector3::One(void){
    return Vector3(1.f,1.f,1.f);
}

inline float Vector3::length() const {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline float Vector3::squared_length() const {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

inline void Vector3::make_unit_vector(){
    float k = 1.0/ sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= k; v[1] *= k; v[2] *= k;
}

inline Vector3 Vector3::operator+(const Vector3 &v1, &const Vector3 &vs){
    return Vector3(v1.v[0] + v.v2[0], v1.v[1] + v.v2[1], v1.v[2] + v.v2[2]);
}

inline Vector3 Vector3::operator-(const Vector3 &v1, &const Vector3 &vs){
    return Vector3(v1.v[0] - v.v2[0], v1.v[1] - v.v2[1], v1.v[2] - v.v2[2]);
}

inline Vector3 Vector3::operator*(const Vector3 &v1, &const Vector3 &vs){
    return Vector3(v1.v[0] * v.v2[0], v1.v[1] * v.v2[1], v1.v[2] * v.v2[2]);
}

inline Vector3 Vector3::operator/(const Vector3 &v1, &const Vector3 &vs){
    return Vector3(v1.v[0] / v.v2[0], v1.v[1] / v.v2[1], v1.v[2] / v.v2[2]);
}

inline Vector3 Vector3::operator*(const Vector3 &v1, float c){
    return Vector3(v1.v[0] * c, v1.v[1] * c, v1.v[2] * c);
}

inline Vector3 Vector3::operator/(const Vector3 &v1, float c){
    return Vector3(v1.v[0] / c, v1.v[1] / c, v1.v[2] / c);
}
