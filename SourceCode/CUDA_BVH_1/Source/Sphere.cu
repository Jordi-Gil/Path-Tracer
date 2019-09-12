#include "Sphere.cuh"

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record &rec) const{
  
  Vector3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  
  if(discriminant > 0){
    float temp = (-b - sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - center) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
    
    temp = (-b + sqrt(b*b-a*c))/a;
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

__device__ bool Sphere::bounding_box (float t0, float t1, aabb &box) const {
	
	t0 = t0;
	t1 = t1;
  
	box = aabb(center - Vector3(radius,radius,radius), center + Vector3(radius,radius,radius));
  
	return true;
  
}