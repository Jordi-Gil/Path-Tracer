#include "Texture.cuh"

__host__ __device__ Texture::Texture(int t, const Vector3 &a, unsigned char *data, int sx, int sy, int _textureIndex, bool _fH, bool _fV, bool _flipUV) {
  
  type = t;
  albedo = a;
  h_image = data;
  nx = sx;
  ny = sy;
  flipHorizontal = _fH;
  flipVertical = _fV;
  flipUV = _flipUV;
  textureIndex = _textureIndex;
}

__host__ __device__ Vector3 Texture::imValue(float u, float v, bool oneTex, unsigned char **d_textures){
  
  u = flipHorizontal ? 1-u : u; 
  v = flipVertical ? 1-v : v;
  
  if(flipUV) {
    float aux = u;
    u = v;
    v = aux;
  }
  
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
    r = int(d_image[3*i*nx + 3*j + 0]) / 255.0f;
    g = int(d_image[3*i*nx + 3*j + 1]) / 255.0f;
    b = int(d_image[3*i*nx + 3*j + 2]) / 255.0f;
  }
  else{
    unsigned char *image = d_textures[textureIndex];
    r = int(image[3*i*nx + 3*j + 0]) / 255.0f;
    g = int(image[3*i*nx + 3*j + 1]) / 255.0f;
    b = int(image[3*i*nx + 3*j + 2]) / 255.0f;
  }
  
  return Vector3(r,g,b);

}

__host__ __device__ Vector3 Texture::value(float u, float v, bool oneTex, unsigned char **d_textures){
  
  if(type == CONSTANT) return albedo;
  else return imValue(u, v, oneTex, d_textures);
  
}

__host__ void Texture::hostToDevice(int numGPUs){
  
  if(type == IMAGE){
  
		cudaSetDevice(numGPUs);
		
    float size = sizeof(unsigned char) * nx * ny * 3;
    cudaMalloc((void **)&d_image, size);
    assert(cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice) == cudaSuccess);
  }
}
