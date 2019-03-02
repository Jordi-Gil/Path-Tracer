#include "Vector3.hh"

Vector3::Vector3(){
    
}
    
Vector3::Vector3(float x, float y, float z){
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

float Vector3::x() const {
    return v[0];
}

float Vector3::y() const {
    return v[1];
}

float Vector3::z() const {
    return v[2];
}

float Vector3::r() const {
    return v[0];
}

float Vector3::g() const {
    return v[1];
}

float Vector3::b() const {
    return v[2];
}

const Vector3& Vector3::operator+() const {
    return *this;
}

Vector3 Vector3::operator-() const {
    return Vector3(-v[0],-v[1],-v[2]);
}

float Vector3::operator[](int i) const {
    return v[i];
}

float& Vector3::operator[](int i){
    return v[i];
}

Vector3 Vector3::Zero(void){
    return Vector3(0.f,0.f,0.f);
}

Vector3 Vector3::One(void){
    return Vector3(1.f,1.f,1.f);
}

float Vector3::length() const {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

float Vector3::squared_length() const {
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

void Vector3::make_unit_vector(){
    float k = 1.0/ sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= k; v[1] *= k; v[2] *= k;
}


float Vector3::dot(const Vector3 &v1, const Vector3 &v2){
    return (v1.v[0] * v2.v[0]) + (v1.v[1] * v2.v[1]) + (v1.v[2] * v2.v[2]);
}

Vector3 Vector3::cross(const Vector3 &v1, const Vector3 &v2){
    return Vector3( 
        (v1.v[1] * v2.v[2] - v1.v[2] * v2.v[1]), 
        (-(v1.v[0] * v2.v[2] - v1.v[2] * v2.v[0])),
        (v1.v[0] * v2.v[1] - v1.v[1] * v2.v[0])
    );
}

Vector3& Vector3::operator+=(const Vector3 &v1){
    v[0] += v1.v[0];
    v[0] += v1.v[1];
    v[0] += v1.v[2];
    return *this;
}

Vector3& Vector3::operator*=(const Vector3 &v1){
    v[0] *= v1.v[0];
    v[0] *= v1.v[1];
    v[0] *= v1.v[2];
    return *this;
}

Vector3& Vector3::operator/=(const Vector3 &v1){
    v[0] /= v1.v[0];
    v[0] /= v1.v[1];
    v[0] /= v1.v[2];
    return *this;
}

Vector3& Vector3::operator-=(const Vector3 &v1){
    v[0] -= v1.v[0];
    v[0] -= v1.v[1];
    v[0] -= v1.v[2];
    return *this;
}

Vector3& Vector3::operator*=(const float t){
    v[0] *= t;
    v[1] *= t;
    v[2] *= t;
    return *this;
}

Vector3& Vector3::operator/=(const float t){
    float k = 1.0/t;
    
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
    return *this;
}

