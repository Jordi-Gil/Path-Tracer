#ifndef _ONB_HH_INCLUDE
#define _ONB_HH_INCLUDE

#include "Vector3.cuh"

class ONB {

public:
  
  __host__ __device__ ONB() {}
  __host__ __device__ inline Vector3 operator[](int i) const {return axis[i];}
  __host__ __device__ Vector3 u();
  __host__ __device__ Vector3 v();
  __host__ __device__ Vector3 w();
  __host__ __device__ Vector3 local(float a, float b, float c);
  __host__ __device__ Vector3 local(const Vector3 &a);
  __host__ __device__ void build_from_w(const Vector3& n);
  
private:
  
  Vector3 axis[3];

};


#endif /* _MATERIAL_HH_INCLUDE */
