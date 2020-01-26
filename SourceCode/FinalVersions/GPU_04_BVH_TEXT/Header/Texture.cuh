#ifndef _TEXTURE_HH_INCLUDE
#define _TEXTURE_HH_INCLUDE

#include "Vector3.cuh"

#include <assert.h>

enum Textype {
  CONSTANT, IMAGE
};

class Texture {

public:
  
  __host__ __device__ Texture() {}
  __host__ __device__ Texture(int t, const Vector3 &a = Vector3::One(), unsigned char *data = 0, int sx = -1, int sy = -1, bool _fH = false, bool _fV = false, bool _flipUV = false);
  
  __host__ __device__ Vector3 value(float u, float v);
  __host__ __device__ Vector3 imValue(float u, float v);
  
  __host__ void hostToDevice();

private:
  
  int type;
  Vector3 albedo;
  unsigned char *h_image;
  unsigned char *d_image;
  int nx;
  int ny;
  bool flipHorizontal, flipVertical, flipUV;
};

#endif /* _TEXTURE_HH_INCLUDE */
