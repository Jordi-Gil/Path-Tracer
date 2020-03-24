#include "ShapeList.cuh"
#include "Triangle.cuh"

__host__ __device__ ShapeList::ShapeList(Triangle* p, int n) {
    
  h_list = p; 
  size = n;
  
  printf("Shape list size: %d\n",size);
    
}

__device__ float ShapeList::pdf_value(const Vector3 &origin, const Vector3 &direction) {
    
  float weight = 1.0/size;
  float sum = 0;
  
  for(int i = 0; i < size; i++) sum += weight*d_list[i].pdf_value(origin, direction);
  
  return sum;
    
}

__device__ Vector3 ShapeList::random(const Vector3 &origin, curandState *_random) {
  
  int index = int(curand_uniform(_random) * size * 0.99999999);
  if(index < 0 || index >= size) printf("Out of index\n");
  return (d_list[index].random(origin, _random));
  
}

__host__ void ShapeList::hostToDevice(int numGPU) {
  
  std::cout <<"Allocating memory for triangles ShapeList" << std::endl;
  
  cudaSetDevice(numGPU);
  
  float _sizen = sizeof(Triangle) * size;
  cudaMalloc((void **)&d_list, _sizen);
  assert(cudaMemcpy(d_list, h_list, _sizen, cudaMemcpyHostToDevice) == cudaSuccess);
  
}
