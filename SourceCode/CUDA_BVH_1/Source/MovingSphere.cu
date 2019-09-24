#include "MovingSphere.cuh"

__host__ __device__ MovingSphere::MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat) {
    center = cen0;
    center1 = cen1; 
    time0 = t0; 
    time1 = t1;
    radius = r; 
    mat_ptr = mat;
    morton_code = 0;
    bounding_box(0,1,box);
}

__host__ __device__ bool MovingSphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const{
  
  Vector3 oc = r.origin() - get_center(r.time());
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;
  
  if(discriminant > 0){
    float temp = (-b - sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - get_center(r.time())) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
    
    temp = (-b + sqrt(b*b-a*c))/a;
    if(temp < t_max && temp > t_min){
      rec.t = temp;
      rec.point = r.point_at_parameter(rec.t);
      rec.normal = (rec.point - get_center(r.time())) / radius;
      rec.mat_ptr = this->mat_ptr;
      return true;
    }
  }
  return false;
}

__host__ __device__ Vector3 MovingSphere::get_center(float time) const{
    return center + ((time-time0) / (time1-time0)) * (center1-center);
}

__host__ __device__ void MovingSphere::bounding_box(float t0, float t1, aabb& box) const {
  	
	aabb box0(get_center(t0) - Vector3(radius, radius, radius), get_center(t0) + Vector3(radius, radius, radius));
	aabb box1(get_center(t1) - Vector3(radius, radius, radius), get_center(t1) + Vector3(radius, radius, radius));
  
	box = surrounding_box(box0, box1); 
}

__host__ __device__ aabb MovingSphere::getBox() const {
    return box;
}

__host__ __device__ unsigned int MovingSphere::getMorton() const {
    return morton_code;
}

__host__ __device__ void MovingSphere::setMorton(unsigned int code) {
    morton_code = code;
}

__host__ __device__ Vector3 MovingSphere::getCenter() const {
    return center;
}
