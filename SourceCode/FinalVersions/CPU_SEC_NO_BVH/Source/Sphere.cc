#include "Sphere.hh"

void get_sphere_uv(const Vector3& p, float& u, float& v) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1-(phi + M_PI) / (2*M_PI);
    v = (theta + M_PI/2) / M_PI;
}

Sphere::Sphere(Vector3 cen, float r, Material mat) {
    center = cen;
    radius = r; 
    mat_ptr = mat;
}

bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) {
  
  Vector3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  
  if(discriminant > 0){
    float temp = (-b - sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      get_sphere_uv((rec.t-center)/radius, rec.u, rec.v);
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
    
    temp = (-b + sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      get_sphere_uv((rec.t-center)/radius, rec.u, rec.v);
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
  }
  return false;
}

Vector3 Sphere::getCenter() {
  return center;
}

float Sphere::getRadius() {
  return radius;
}

Material Sphere::getMaterial() {
  return mat_ptr;
}
