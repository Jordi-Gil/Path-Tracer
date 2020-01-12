#include "Texture.cuh"

__host__ __device__ Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy){
  type = t;
  albedo = a;
  image = data;
  nx = sx;
  ny = sy;
}

__host__ __device__ Vector3 Texture::imValue(float u, float v, Vector3 p){
  
//   int i = (  u)*nx;
//   int j = (1-v)*ny-0.001;
//   int j = (1-v)*ny;
  
  /*
  Vector3 red(1,0,0);
  Vector3 black(0,0,0);
  Vector3 yellow(1,1,0);
  
  Vector3 color1 = (red * (1-u)) + (yellow * u);
  Vector3 color2 = (red * (1-v)) + (yellow * v);
  */
  
  
//   if(i < 0) i = 0;
//   if(j < 0) j = 0;
//   
//   if(i > nx-1) i = nx-1;
//   if(j > ny-1) j = ny-1;
  
//   float r = int(image[0]) / 255.0f; //int(image2[3*i*nx + 3*j + 0]) / 255.0f;
//   float g = int(image[1]) / 255.0f; //int(image2[3*i*nx + 3*j + 1]) / 255.0f;
//   float b = int(image[2]) / 255.0f; //int(image2[3*i*nx + 3*j + 2]) / 255.0f;
//   
/*   return (color1 + color2)/2; */
//   return Vector3(r,g,b); 

  Vector3 b(u,v,1-u-v);
  
  //printf("(%f,%f,%f) - (%f,%f,%f)\n",b[0],b[1],b[2],p[0],p[1],p[2]);

  float z = b.x()/p.z() + b.y()/p.z() + b.z()/p.z();

  float uu = (b.x()*u/p.z() + b.y()*u/p.z() + b.z()*u/p.z()) / z;
  float vv = (b.x()*v/p.z() + b.y()*v/p.z() + b.z()*v/p.z()) / z;

  Vector3 white(1,1,1);
  Vector3 red(1,0,0);
  
  int tx = (int) 10*uu;
  int ty = (int) 10*vv;
  
  int oddity = (tx & 0x01) == (ty & 0x01);
  
  int edge = ((10 * uu - tx < 0.1) && oddity) || (10 * vv - ty < 0.1);
  
  return edge ? white : red;

}

__host__ __device__ Vector3 Texture::value(float u, float v, Vector3 p){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u,v,p);
  
}
