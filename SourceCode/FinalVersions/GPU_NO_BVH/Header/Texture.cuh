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
  __host__ __device__ Texture(int t, const Vector3 &a = Vector3::One(), unsigned char *data = 0, int sx = -1, int sy = -1, int _textureIndex = -1, bool _fH = false, bool _fV = false, bool _flipUV = false);
  
  __host__ __device__ Vector3 value(float u, float v, bool oneTex = false, unsigned char **d_textures = 0);
  __host__ __device__ Vector3 imValue(float u, float v, bool oneTex, unsigned char **d_textures);
  
  __host__ void hostToDevice(int numGPUs);

private:
  
  int type;
  Vector3 albedo;
  unsigned char *h_image;
  unsigned char *d_image;
  int nx;
  int ny;
  bool flipHorizontal, flipVertical, flipUV;
  int textureIndex;
};

#endif /* _TEXTURE_HH_INCLUDE */
