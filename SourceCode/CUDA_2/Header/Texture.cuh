#ifndef _TEXTURE_HH_INCLUDE
#define _TEXTURE_HH_INCLUDE

#include "Vector3.cuh"

enum Textype {
  CONSTANT, IMAGE
};

class Texture {

public:
  
  __host__ __device__ Texture() {}
  __host__ __device__ Texture(int t, const Vector3 &a = Vector3::One(), unsigned char *data = 0, int sx = -1, int sy = -1);
  
  __host__ __device__ Vector3 value(float u, float v);
  __host__ __device__ Vector3 imValue(float u, float v);

private:
  
  int type;
  Vector3 albedo;
  unsigned char *image;
  int nx;
  int ny;
  
};

#endif /* _TEXTURE_HH_INCLUDE */
