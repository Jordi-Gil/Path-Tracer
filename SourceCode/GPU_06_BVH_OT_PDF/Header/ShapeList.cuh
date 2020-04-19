#ifndef _SHAPE_LIST_SHAPE_HH_INCLUDE
#define _SHAPE_LIST_SHAPE_HH_INCLUDE

#include <curand.h>
#include <curand_kernel.h>

class Triangle;
class Vector3;

class ShapeList {

public:
  
  __host__ __device__ ShapeList() {};
  __host__ __device__ ShapeList(Triangle* p, int n);
  
  __device__ float pdf_value(const Vector3 &origin, const Vector3 &direction);
  
  __device__ Vector3 random(const Vector3 &origin, curandState *_random);
  
  __host__ void hostToDevice(int numGPU);
  
private:
  
  Triangle *h_list;
  Triangle *d_list;
  int size;
  
};

#endif /* _SHAPE_LIST_SHAPE_HH_INCLUDE */
