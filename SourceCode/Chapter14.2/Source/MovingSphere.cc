#include "MovingSphere.hh"

bool MovingSphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const{
  
  Vector3 oc = r.origin() - center(r.time());
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  
  if(discriminant > 0){
    float temp = (-b - sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center(r.time())) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
    
    temp = (-b + sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center(r.time())) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
  }
  return false;
}

Vector3 MovingSphere::center(float time) const{
    return center0 + ((time-time0) / (time1-time0)) * (center1-center0);
}

bool MovingSphere::bounding_box(float t0, float t1, aabb& box) const {
  
  aabb box0(center(t0) - Vector3(radius, radius, radius), center(t0) + Vector3(radius, radius, radius));
  aabb box1(center(t1) - Vector3(radius, radius, radius), center(t1) + Vector3(radius, radius, radius));
  
  box = surrounding_box(box0, box1);
  
  return true;
  
}
