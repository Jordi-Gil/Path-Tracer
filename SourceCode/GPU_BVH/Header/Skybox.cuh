#ifndef _SKYBOX_HH_INCLUDE
#define _SKYBOX_HH_INCLUDE

#include "Material.cuh"

enum Side{
  LEFT, RIGHT, TOP, BOTTOM, FRONT, BACK
};

class Rectangle {

public:
  
  __host__ __device__ Rectangle(){}
  __host__ __device__ Rectangle(float _a0, float _a1, float _b0, float _b1, float _k, Material m, int _t, bool _flip = false) : a0(_a0), a1(_a1), b0(_b0), b1(_b1), k(_k), mat(m), type(_t), flipN(_flip) {};
  
  __device__ bool hitXY(const Ray& r, float t_min, float t_max, hit_record& rec){
    
    float t = (k-r.origin().z()) / r.direction().z();
    if(t < t_min || t > t_max) return false;
    
    float x = r.origin().x() + t*r.direction().x();
    float y = r.origin().y() + t*r.direction().y();
    
    if(x < a0 || x > a1 || y < b0 || y > b1) return false;
    
    rec.u = (x-a0)/(a1-a0);
    rec.v = (y-b0)/(b1-b0);
    rec.t = t;
    rec.mat_ptr = mat;
    rec.point = r.point_at_parameter(t);
    rec.normal = Vector3(0, 0, 1);
    if(flipN) rec.normal = -rec.normal;
    
    return true;
    
  }
  
  __device__ bool hitXZ(const Ray& r, float t_min, float t_max, hit_record& rec){
    
    float t = (k-r.origin().y()) / r.direction().y();
    if(t < t_min || t > t_max) return false;
    
    float x = r.origin().x() + t*r.direction().x();
    float z = r.origin().z() + t*r.direction().z();
    
    if(x < a0 || x > a1 || z < b0 || z > b1) return false;
    
    rec.u = (x-a0)/(a1-a0);
    rec.v = (z-b0)/(b1-b0);
    rec.t = t;
    rec.mat_ptr = mat;
    rec.point = r.point_at_parameter(t);
    rec.normal = Vector3(0, 1, 0);
    if(flipN) rec.normal = -rec.normal;
    
    return true;
    
  }
  
  __device__ bool hitYZ(const Ray& r, float t_min, float t_max, hit_record& rec){
    
    float t = (k-r.origin().x()) / r.direction().x();
    if(t < t_min || t > t_max) return false;
    
    float y = r.origin().y() + t*r.direction().y();
    float z = r.origin().z() + t*r.direction().z();
    
    if(y < a0 || y > a1 || z < b0 || z > b1) return false;
    
    rec.u = (y-a0)/(a1-a0);
    rec.v = (z-b0)/(b1-b0);
    rec.t = t;
    rec.mat_ptr = mat;
    rec.point = r.point_at_parameter(t);
    rec.normal = Vector3(1, 0, 0);
    if(flipN) rec.normal = -rec.normal;
    
    return true;
    
  }
  
  __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec){
    
    if(type == FRONT || type == BACK) return hitXY(r, t_min, t_max, rec);
    else if(type == TOP || type == BOTTOM) return hitXZ(r, t_min, t_max, rec);
    else return hitYZ(r, t_min, t_max, rec);
    
  }
  
  __host__ void hostToDevice(int numGPUs){ mat.hostToDevice(numGPUs); }
  
private:
  
  float a0, a1, b0, b1, k, dim1, dim2, invdim1, invdim2;
  Material mat;
  int type;
  bool flipN;
  
};

class Skybox {
  
public:
  
  Skybox(){}
  Skybox(Vector3 a, Vector3 b, const std::string &dir);
  void load(const std::string &dir);
  __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  __host__ void hostToDevice(int numGPUs){ 
    for(int i = 0; i < 6; i++) list[i].hostToDevice(numGPUs);
  }
  
private:
  
  Vector3 bottomLeft, topRight;
  
  Rectangle list[6];
  
};

#endif /* _SKYBOX_HH_INCLUDE */
