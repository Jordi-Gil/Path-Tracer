#include "Sphere.cuh"

__device__ void get_sphere_uv(const Vector3& p, float& u, float& v) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1-(phi + M_PI) / (2*M_PI);
    v = (theta + M_PI/2) / M_PI;
}

__host__ __device__ Sphere::Sphere(Vector3 cen, float r, Material mat) { 
  center = cen;
  radius = r;
  mat_ptr = mat;
  morton_code = 0;
  bounding_box(box);
}

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record &rec) {
  
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

__host__ __device__ void Sphere::bounding_box(aabb& box) {
  
  box = aabb(center - Vector3(radius), center + Vector3(radius));
  
}

__host__ __device__ aabb Sphere::getBox() {
    return box;
}

__host__ __device__ long long Sphere::getMorton() {
    return morton_code;
}

__host__ __device__ void Sphere::setMorton(long long code) {
    morton_code = code;
}

__host__ __device__ Vector3 Sphere::getCenter() {
    return center;
}

__host__ __device__ float Sphere::getRadius() {
    return radius;
}

__host__ __device__ Material Sphere::getMaterial() {
    return mat_ptr;
}

