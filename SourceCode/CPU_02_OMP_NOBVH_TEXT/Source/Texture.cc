#include "Texture.hh"

Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy, bool _fH, bool _fV, bool _flipUV){
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
  flipHorizontal = _fH;
  flipVertical = _fV;
  flipUV = _flipUV;
}

Vector3 Texture::imValue(float u, float v){
  
  u = flipHorizontal ? 1-u : u; 
  v = flipVertical ? 1-v : v;
  
  if(flipUV) std::swap(u,v);
  
  int i = u * (nx-1);
  int j = v * (ny-1);
  
  
  if(i < 0) i = 0;
  if(j < 0) j = 0;
  
  if(i > nx-1) i = nx-1;
  if(j > ny-1) j = ny-1;
  
  float r = int(image[3*i + 3*nx*j + 0]) / 255.0f;
  float g = int(image[3*i + 3*nx*j + 1]) / 255.0f;
  float b = int(image[3*i + 3*nx*j + 2]) / 255.0f;
  
  return Vector3(r,g,b);
}

Vector3 Texture::value(float u, float v){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u,v);
  
}
