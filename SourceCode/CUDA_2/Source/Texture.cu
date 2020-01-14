#include "Texture.cuh"

__host__ __device__ Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy){
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
}

__host__ __device__ Vector3 Texture::imValue(float u, float v, const Vector3 vertex[3], const Vector3 uv[3]){
  
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

  Vector3 P(u,v,1-u-v);
  
  Vector3 A = vertex[0]; Vector3 B = vertex[1]; Vector3 C = vertex[2];
  
  float baryA = ( (B.y()-C.y())*(P.x()-C.x()) + (C.x()-B.x())*(P.y()-C.y()) ) / ( (B.y()-C.y())*(A.x()-C.x()) + (C.x()-B.x())*(P.y()-C.y()) );
  
  float baryB = ( (C.y()-A.y())*(P.x()-C.x()) + (A.x()-C.x())*(P.y()-C.y()) ) / ( (B.y()-C.y())*(A.x()-C.x()) + (C.x()-B.x())*(A.y()-C.y()) );
  float baryC = 1 - baryA - baryB;
  
  Vector3 Puv = baryA*uv[0] + baryB*uv[1] + baryC*uv[2];
  
  float uu = Puv[0];
  float vv = Puv[1];

  Vector3 white(1,1,1);
  Vector3 red(1,0,0);
  
  int tx = (int) 10*uu;
  int ty = (int) 10*vv;
  
  int oddity = (tx & 0x01) == (ty & 0x01);
  
  int edge = ((10 * uu - tx < 0.1) && oddity) || (10 * vv - ty < 0.1);
  
  return edge ? white : red;

}

__host__ __device__ Vector3 Texture::value(float u, float v, const Vector3 vertex[3], const Vector3 uv[3]){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u, v, vertex, uv);
  
}
