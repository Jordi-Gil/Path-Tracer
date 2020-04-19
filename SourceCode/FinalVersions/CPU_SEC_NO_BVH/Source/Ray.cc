#include "Ray.hh"

Ray::Ray(){}

Ray::Ray(const Vector3& a, const Vector3& b, float ti){
    A = a;
    B = b;
    _time = ti;
}

Vector3 Ray::origin() const{
    return A;
}

Vector3 Ray::direction() const{
    return B;
}

float Ray::time() const{
    return _time;
}

Vector3 Ray::point_at_parameter(float t) const{
    return A + t*B;
}
