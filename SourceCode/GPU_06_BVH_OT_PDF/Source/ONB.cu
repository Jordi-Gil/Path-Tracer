#include "ONB.cuh"

__host__ __device__ Vector3 ONB::u() { return axis[0];}

__host__ __device__ Vector3 ONB::v() { return axis[1];}

__host__ __device__ Vector3 ONB::w() { return axis[2];}

__host__ __device__ Vector3 ONB::local(float a, float b, float c) {
    
  return a*u() + b*v() + c*w();
    
}
  
__host__ __device__ Vector3 ONB::local(const Vector3 &a) {
  
  Vector3 b = a.x()*u() + a.y()*v() + a.z()*w();
  
  return b;
    
}
  
__host__ __device__ void ONB::build_from_w(const Vector3& n){
  axis[2] = unit_vector(n);
  Vector3 a;
    
  if(fabs(w().x()) > 0.9) a = Vector3(0,1,0);
  else a = Vector3(1,0,0);
  
  axis[1] = unit_vector(cross(w(),a));
  axis[0] = cross(w(),v());
}
