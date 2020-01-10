#include "Texture.hh"

Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy){
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
}

Vector3 Texture::imValue(float u, float v){
  int i = (  u)*nx;
  int j = (1-v)*ny-0.0001;
  
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
