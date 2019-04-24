#include <iostream>
#include <cfloat>
#include <ctime>
#include "Ray.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
    if(result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << std::endl;
        std::cerr << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const Vector3& center, float radius, const Ray& ray){
    Vector3 oc = ray.origin() - center;
    float a = dot(ray.direction(), ray.direction());
    float b = 2.0 * dot(oc, ray.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    return (discriminant > 0);
}

__device__ Vector3 color(const Ray& ray){
  
  if(hit_sphere(Vector3(0, 0, -1), 0.5, ray)){
    return Vector3(1, 0, 0);
  }
  
  Vector3 unit_direction = unit_vector(ray.direction());
  float t = 0.5*(unit_direction.y() + 1.0);
  return (1.0-t) * Vector3(1.0,1.0,1.0) + t*Vector3(0.5, 0.7, 1.0);
}

__global__ void render(Vector3 *fb, int max_x, int max_y, Vector3 lower_left_corner, Vector3 horizontal, Vector3 vertical, Vector3 origin){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if( (i < max_x) && (j < max_y) ){
    int pixel_index = j*max_x + i;
    
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    
    Ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    
    fb[pixel_index] = color(r);
    
  }
}

int main()
{
  
  cudaEvent_t E0, E1;
  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  
  float GPUtime;

  int nx = 2000;
  int ny = 1000;
  

  int num_pixels = nx*ny;
  float fb_size = num_pixels*sizeof(Vector3); // frame buffer, RGB para cada pixel
  
  Vector3 *frameBuffer;
  checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fb_size)); //Reservamos memoria
  
  int tx = 8;
  int ty = 8;
  
  dim3 blocks(nx/tx+1, ny/ty+1); // Dividmos el trabajo en 8x8 threads
  dim3 threads(tx,ty);
  
  Vector3 lower_left_corner(-2.0, -1.0, -1.0);
  Vector3 horizontal(4.0, 0.0, 0.0);
  Vector3 vertical(0.0, 2.0, 0.0);
  Vector3 origin(0.0, 0.0, 0.0);

  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  render<<<blocks, threads>>>(frameBuffer, nx, ny, lower_left_corner, horizontal, vertical, origin);
  
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);
  
  cudaEventElapsedTime(&time,E0,E1);
  cudaEventDestroy(E0); cudaEventDestroy(E1);
  
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  
  for(int j = ny-1; j >= 0; j--){
    for(int i = 0; i < nx; i++){

      size_t pixel_index = j*nx + i;
      
      Vector3 col = frameBuffer[pixel_index];

      float r = col.x();
      float g = col.y();
      float b = col.z();
      
      int ir = int(255.99*r);
      int ig = int(255.99*g);
      int ib = int(255.99*b);
      
      std::cout << ir << " " << ig << " " << ib << std::endl;
    }
  }
  checkCudaErrors(cudaFree(frameBuffer));
  std::cerr << "GPU Time: " << GPUtime << " milisegs."<< std::endl;
  
}
