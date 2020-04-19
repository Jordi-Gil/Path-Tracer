#include "Texture.hh"

Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy, int _textureIndex, bool _fH, bool _fV, bool _flipUV) {
  
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
  flipHorizontal = _fH;
  flipVertical = _fV;
  flipUV = _flipUV;
  textureIndex = _textureIndex;
}

Vector3 Texture::imValue(float u, float v, bool oneTex, unsigned char **textures){
  
  u = flipHorizontal ? 1-u : u; 
  v = flipVertical ? 1-v : v;
  
  if(flipUV) std::swap(u,v);
  
  int i = u * (nx-1);
  int j = v * (ny-1);
  
  if(i < 0) i = 0;
  if(j < 0) j = 0;
  
  if(i > nx-1) i = nx-1;
  if(j > ny-1) j = ny-1;
  
  float r;
  float g;
  float b;
  
  if(!oneTex or textureIndex == 999){
    r = int(image[3*i*nx + 3*j + 0]) / 255.0f;
    g = int(image[3*i*nx + 3*j + 1]) / 255.0f;
    b = int(image[3*i*nx + 3*j + 2]) / 255.0f;
  }
  else{
    unsigned char *aux_im = textures[textureIndex];
    r = int(aux_im[3*i*nx + 3*j + 0]) / 255.0f;
    g = int(aux_im[3*i*nx + 3*j + 1]) / 255.0f;
    b = int(aux_im[3*i*nx + 3*j + 2]) / 255.0f;
  }
  
  return Vector3(r,g,b);
}

Vector3 Texture::value(float u, float v, bool oneTex, unsigned char **textures){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u, v, oneTex, textures);
  
}
