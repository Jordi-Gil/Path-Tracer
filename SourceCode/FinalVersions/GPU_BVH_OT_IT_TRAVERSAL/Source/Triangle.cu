#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 uv1, Vector3 uv2, Vector3 uv3) {
    vertex[0] = v1;
    vertex[1] = v2;
    vertex[2] = v3;
    centroid = (v1+v2+v3)/3;
    mat_ptr = mat;
    morton_code = 0;
    bounding_box(box);
    uv[0] = uv1;
    uv[1] = uv2;
    uv[2] = uv3;
}

__host__ __device__ bool Triangle::hit(const Ray& r, float t_min, float t_max, hit_record& rec) {
  
  Vector3 E1 = vertex[1] - vertex[0];
  Vector3 E2 = vertex[2] - vertex[0];
  Vector3 T = r.origin() - vertex[0];

  Vector3 P = cross(r.direction(), E2);
  Vector3 Q = cross(T, E1);
  
  float determinant = dot(P,E1);
  float invDet = 1.0f / determinant;
  
  float t = dot(Q, E2) * invDet;
  float u = dot(P, T) * invDet;
  float v = dot(Q, r.direction()) * invDet;
  
  
  if(determinant > -t_min and determinant < t_min) 
    return false;
  
  if(u < 0.0f || u > 1.0f) return false;
  
  if(v < 0.0f || u + v > 1.0f) return false;
  
  if(t > t_min && t < t_max) {
    rec.t = t;
    Vector3 aux = (1-u-v)*uv[0] + u*uv[1] + v*uv[2];
    rec.u = aux[0];
    rec.v = aux[1];
    rec.point = r.point_at_parameter(rec.t);
    rec.normal = normalize(cross(E1, E2));
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
  
  if(x_max == x_min) { x_max += 0.00005; x_min -= 0.00005; }
  if(y_max == y_min) { y_max += 0.00005; y_min -= 0.00005; }
  if(z_max == z_min) { z_max += 0.00005; z_min -= 0.00005; }
  
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
