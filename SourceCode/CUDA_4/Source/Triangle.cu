#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 t) {
    vertex[0] = v1;
    vertex[1] = v2;
    vertex[2] = v3;
    centroid = (v1+v2+v3)/3;
    mat_ptr = mat;
    morton_code = 0;
    bounding_box(box);
    uv = t;
}

__host__ __device__ bool Triangle::hit(const Ray& r, float t_min, float t_max, hit_record& rec) {
  
  Vector3 e1 = vertex[1] - vertex[0];
  Vector3 e2 = vertex[2] - vertex[0];
  
  Vector3 P = cross(r.direction(), e2);
  float determinant = dot(e1, P);
  
  if(determinant > -t_min and determinant < t_min) 
    return false;
  
  float invDet = 1.0f / determinant;
  
  Vector3 T = r.origin() - vertex[0];
  float u = dot(T, P) * invDet;
  
  if(u < 0.0f || u > 1.0f) return false;
  
  Vector3 Q = cross(T, e1);
  float v = dot(r.direction(), Q) * invDet;
  
  if(v < 0.0f || u + v > 1.0f) return false;
  
  float temp = dot(e2, Q) * invDet;
  if(temp > t_min && temp < t_max) {
    rec.t = temp;
    rec.point = r.point_at_parameter(rec.t);
    rec.normal = normalize(cross(e1, e2));
    rec.mat_ptr = this->mat_ptr;
    
    return true;
  }
  
  return false;
}

__host__ __device__ void Triangle::bounding_box(aabb& box) {
  
	float x_max = math::max(math::max(vertex[0].x(),vertex[1].x()),vertex[2].x());
	float y_max = math::max(math::max(vertex[0].y(),vertex[1].y()),vertex[2].y());
	float z_max = math::max(math::max(vertex[0].z(),vertex[1].z()),vertex[2].z());
	
	float x_min = math::min(math::min(vertex[0].x(),vertex[1].x()),vertex[2].x());
	float y_min = math::min(math::min(vertex[0].y(),vertex[1].y()),vertex[2].y());
	float z_min = math::min(math::min(vertex[0].z(),vertex[1].z()),vertex[2].z());
  
  if(x_max == x_min) { x_max += 0.0005; x_min -= 0.0005; }
  if(y_max == y_min) { y_max += 0.0005; y_min -= 0.0005; }
  if(z_max == z_min) { z_max += 0.0005; z_min -= 0.0005; }
  
  Vector3 max(x_max, y_max, z_max), min(x_min, y_min, z_min);
	
	box = aabb(min,max);
  
}

__host__ __device__ aabb Triangle::getBox() {
  return box;
}

__host__ __device__ long long Triangle::getMorton() {
  return morton_code;
}

__host__ __device__ void Triangle::setMorton(long long code) { 
  morton_code = code;
}

__host__ __device__ Vector3 Triangle::operator[](int i) const {
  if(i < 0 && i > 2) assert(0);
  return vertex[i];
}

__host__ __device__ Vector3& Triangle::operator[](int i) {
  if(i < 0 && i > 2) assert(0);
  return vertex[i];
}

__host__ __device__ Vector3 Triangle::getCentroid() {
  return centroid;
}

__host__ __device__ Material Triangle::getMaterial() {
  return mat_ptr;
}

__host__ __device__ void Triangle::resizeBoundingBox() {
  bounding_box(box);
}
