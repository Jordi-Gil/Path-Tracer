#include "Texture.cuh"

__host__ __device__ Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy){
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
}

__host__ __device__ Vector3 Texture::imValue(float u, float v){
  
//   int i = (  u)*nx;
//   int j = (1-v)*ny-0.001;
//   int j = (1-v)*ny;
  
//   if(i < 0) i = 0;
//   if(j < 0) j = 0;
//   
//   if(i > nx-1) i = nx-1;
//   if(j > ny-1) j = ny-1;
  
//   float r = int(image[0]) / 255.0f; //int(image2[3*i*nx + 3*j + 0]) / 255.0f;
//   float g = int(image[1]) / 255.0f; //int(image2[3*i*nx + 3*j + 1]) / 255.0f;
//   float b = int(image[2]) / 255.0f; //int(image2[3*i*nx + 3*j + 2]) / 255.0f;
//   

//   return Vector3(r,g,b); 

  Vector3 white(1,1,1);
  Vector3 red(1,0,0);
  
  int tx = (int) 10*u;
  int ty = (int) 10*v;
  
  int oddity = (tx & 0x01) == (ty & 0x01);
  
  int edge = ((10 * u - tx < 0.1) && oddity) || (10 * v - ty < 0.1);
  
  return edge ? white : red;

}

__host__ __device__ Vector3 Texture::value(float u, float v){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u,v);
  
}
