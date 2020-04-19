#ifndef _PDF_SHAPE_HH_INCLUDE
#define _PDF_SHAPE_HH_INCLUDE

#include <curand.h>
#include <curand_kernel.h>

#include "ONB.cuh"

#include "ShapeList.cuh"

enum type_pdf {
  COSINE, SHAPE
};

class pdf{
  
  
public:

  __host__ __device__ pdf() {};
  __host__ __device__ pdf(int _type, const Vector3 &w = Vector3::Zero(), ShapeList* p = 0, const Vector3 &o = Vector3::Zero());
  
  __device__ float value(const Vector3 &direction);
  
  __device__ float cosine_value(const Vector3 &direction);
  __device__ float shape_value(const Vector3 &direction);
  
  __device__ Vector3 generate(curandState *_random);
  
  __device__ Vector3 generate_cosine(curandState *_random);
  __device__ Vector3 generate_shape(curandState *_random);
  
private:
  
  int type;
  ONB uvw;
  Vector3 origin;
  ShapeList *ptr;
  
};


#endif /* _PDF_SHAPE_HH_INCLUDE */
