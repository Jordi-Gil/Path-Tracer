#include "Triangle.cuh"

__host__ __device__ Triangle::Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 uv1, Vector3 uv2, Vector3 uv3) {
    vertex[0] = v1;
    vertex[1] = v2;
    vertex[2] = v3;
    centroid = (v1+v2+v3)/3;
    mat_ptr = mat;
    uv[0] = uv1;
    uv[1] = uv2;
    uv[2] = uv3;
    
    //printf("(%f,%f,%f) - (%f,%f,%f) - (%f,%f,%f)\n", uv1[0],uv1[2],uv1[3],uv2[0],uv2[1],uv2[2],uv3[0],uv3[1],uv3[2]);
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
    rec.u = u;
    rec.v = u;
    rec.vertex[0] = vertex[0]; rec.vertex[1] = vertex[1]; rec.vertex[2] = vertex[2];
    rec.uv[0] = uv[0]; rec.uv[1] = uv[1]; rec.uv[1] = uv[2];
    rec.point = r.point_at_parameter(rec.t);
    rec.normal = normalize(cross(e1, e2));
    rec.mat_ptr = this->mat_ptr;
    
    return true;
  }
  
  return false;
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
