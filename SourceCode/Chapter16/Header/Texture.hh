#ifndef _TEXTURE_HH_INCLUDE
#define _TEXTURE_HH_INCLUDE

#include "Vector3.hh"

enum Textype {
  CONSTANT, IMAGE
};

class Texture {

public:
    
  Texture() {}
  Texture(int t, const Vector3 &a = Vector3::One(), unsigned char *data = 0, int sx = -1, int sy = -1, bool _fH = false, bool _fV = false, bool _flipUV = false);
  
  Vector3 value(float u, float v);
  Vector3 imValue(float u, float v);
  
private:  
  
  int type;
  Vector3 albedo;
  unsigned char *image;
  int nx;
  int ny;
  bool flipHorizontal, flipVertical, flipUV;
};


#endif /* _TEXTURE_HH_INCLUDE */
