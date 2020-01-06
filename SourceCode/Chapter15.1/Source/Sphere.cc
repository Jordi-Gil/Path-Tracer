#include "Sphere.hh"

Sphere::Sphere(Vector3 cen, float r, Material mat) {
    center = cen;
    radius = r; 
    mat_ptr = mat;
    morton_code = 0;
    bounding_box(box);
}

bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) {
  
  Vector3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  
  if(discriminant > 0){
    float temp = (-b - sqrt(b*b - a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
    
    temp = (-b + sqrt(b*b - a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
  }
  return false;
}

void Sphere::bounding_box(aabb& box) {
  
  box = aabb(center - Vector3(radius), center + Vector3(radius));
  
}

aabb Sphere::getBox() {
  return box;
}

long long Sphere::getMorton() {
  return morton_code;
}

void Sphere::setMorton(long long code) { 
  morton_code = code;
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