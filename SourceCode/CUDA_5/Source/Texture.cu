#include "Texture.cuh"

__host__ __device__ Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy){
  type = t;
  albedo = a;
  h_image = data;
  nx = sx;
  ny = sy;
}

__host__ __device__ Vector3 Texture::imValue(float u, float v){
  
  int i = (  u)*nx;
  int j = (  v)*ny;
  
  if(i < 0) i = 0;
  if(j < 0) j = 0;
  
  if(i > nx-1) i = nx-1;
  if(j > ny-1) j = ny-1;
  
  float r = int(d_image[3*i*nx + 3*j + 0]) / 255.0f;
  float g = int(d_image[3*i*nx + 3*j + 1]) / 255.0f;
  float b = int(d_image[3*i*nx + 3*j + 2]) / 255.0f;
  
  return Vector3(r,g,b);

}

__host__ __device__ Vector3 Texture::value(float u, float v){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u,v);
  
}

__host__ void Texture::hostToDevice(){
  
  if(type == IMAGE){
  
    float size = sizeof(unsigned char) * nx * ny * 3;
    cudaMalloc((void **)&d_image, size);
    assert(cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice) == cudaSuccess);
  }
}
