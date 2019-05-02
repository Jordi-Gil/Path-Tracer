#include <iostream>
#include <cfloat>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#include "Sphere.cuh"
#include "HitableList.cuh"
#include "Camera.cuh"
#include "Material.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

void error(const char *message){
  
  std::cerr << message << std::endl;
  exit(0);
}


void help(){

  std::cerr << "\n"  << std::endl;
  std::cerr << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
  std::cerr << "\t                size: (1200x600) | AAit: 10 | depth: 50 | spheres: 11 | nthreads: 8"  << std::endl;
  std::cerr << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cerr << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cerr << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cerr << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cerr << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cerr << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cerr << "\n" << std::endl;
  std::cerr << "Examples of usage:" << std::endl;
  std::cerr << "./ray_tracing -d"  << std::endl;
  std::cerr << "./ray_tracing -nthreads 16 -sizeX 2000"<< std::endl;
  exit(0);
  
}


void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 1200; ny = 600; ns = 10; depth = 50; dist = 11; nthreads = 8;
  
  bool v_default = false;
  
  for(int i = 1; i < argc; i += 2){
    
    if(v_default) error("Error usage. Use [-h] [--help] to see the usage.");
    
    switch(str2int(argv[i]))
    {
      case str2int("-d"):
      case str2int("--default"):
        if((i+1) < argc) error("The default parameter cannot have more arguments.");
        v_default = true;
        break;
      
      case str2int("-sizeX"):
        if((i+1) >= argc) error("-sizeX value expected");
        nx = atoi(argv[i+1]);
        if(nx == 0) error("-sizeX value expected or cannot be 0");
        break;
      
      case str2int("-sizeY"):
        if((i+1) >= argc) error("-sizeY value expected");
        ny = atoi(argv[i+1]);
        if(ny == 0) error("-sizeY value expected or cannot be 0");
        break;
          
      case str2int("-AAit"):
        if((i+1) >= argc) error("-AAit value expected");
        ns = atoi(argv[i+1]);
        if(ns == 0) error("-AAit value expected or cannot be 0");
        break;

      case str2int("-depth"):
        if((i+1) >= argc) error("-depth value expected");
        depth = atoi(argv[i+1]);
        if(depth == 0) error("-depth value expected or cannot be 0");
        break;
      
      case str2int("-spheres"):
        if((i+1) >= argc) error("-spheres value expected");
        depth = atoi(argv[i+1]);
        if(depth == 0) error("-spheres value expected or cannot be 0");
        break;
      case str2int("-nthreads"):
        if((i+1) >= argc) error("-nthreads value expected");
        nthreads = atoi(argv[i+1]);
        if(nthreads == 0) error("-nthreads value expected or cannot be 0");
        break;
      case str2int("-h"):
      case str2int("-help"):
        help();
      default:
        error("Error usage. Use [-h] [--help] to see the usage.");
        break;
    }
  }
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
  if(result){
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << std::endl;
    std::cerr << cudaGetErrorString(result) << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}

__global__ void free_world(Hitable **d_list, Hitable **d_world, Camera **d_cam, int n) {
  for(int i = 0; i < n; i++){
    delete ((Sphere *)d_list[i])->mat_ptr;
    delete d_list[i];
  }
  delete *d_world;
  delete *d_cam;
}

#define Random (curand_uniform(&local_random))

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_cam, int nx, int ny, int dist, int n, curandState *random){
  
  if(threadIdx.x == 0 && blockIdx.x == 0){
    
    curandState local_random = *random;
    
    d_list[0] = new Sphere(Vector3(0,-1000,-1), 1000, 
                           new Lambertian(Vector3(0.5, 0.5, 0.5)));
    
    int i = 1;
    for(int a = -dist; a < dist; a++){
      for(int b = -dist; b < dist; b++){
        float material = Random;
        Vector3 center(a+Random, 0.2, b+Random);
        
        if(material < 0.8f){
          d_list[i++] = new Sphere(center, 0.2, 
                                   new Lambertian(Vector3(Random*Random, 
                                                          Random*Random, 
                                                          Random*Random)));
        }
        else if(material < 0.95f){
          d_list[i++] = new Sphere(center, 0.2, 
                                   new Metal(Vector3(0.5f*(1.0+Random), 
                                                     0.5f*(1.0+Random), 
                                                     0.5f*(1.0+Random)),0.5*Random));
        }
        else{
          d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
        }
      }
    }
    
    d_list[i++] = new Sphere(Vector3( 0, 1, 0), 1.0, new Dielectric(1.5));
    d_list[i++] = new Sphere(Vector3(-4, 1, 0), 1.0, new Lambertian(Vector3(0.4, 0.2, 0.1)));;
    d_list[i++] = new Sphere(Vector3( 4, 1, 0), 1.0, new Metal(Vector3(0.7, 0.6, 0.5),0.0));
    
    *random = local_random;
    
    *d_world = new HitableList(d_list,n);
    
    Vector3 lookfrom(13,2,3);
    Vector3 lookat(0,0,0);
    Vector3 up(0,1,0);
    float dist_to_focus = 10; (lookfrom-lookat).length();
    float aperture = 0.1;
    
    *d_cam = new Camera(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus);
  }
  
}

__device__ Vector3 color(const Ray& ray, Hitable **world, int depth, curandState *random){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3(1.0,1.0,1.0);
  for(int i = 0; i < depth; i++){
    hit_record rec;
    if( (*world)->hit(cur_ray, 0.001, FLT_MAX, rec)){
      
      Ray scattered;
      Vector3 attenuation;
      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, random)){
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else return Vector3(0.0,0.0,0.0);
    }
    else{
      Vector3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5*(unit_direction.y() + 1.0);
      Vector3 c = (1.0 - t)*Vector3(1.0,1.0,1.0) + t*Vector3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return Vector3(0.0,0.0,0.0);
}

__global__ void rand_init(curandState *random, int seed){
  if(threadIdx.x == 0 && blockIdx.x == 0){
    curand_init(seed, 0, 0, random);
  }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state,unsigned long long seed){
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if( (i >= max_x) || (j >= max_y) ) return;
    
  int pixel_index = j*max_x + i;
    
  curand_init((seed << 20) + pixel_index, 0, 0, &rand_state[pixel_index]);
  
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Hitable **world, curandState *d_rand_state, int depth){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  curandState random;
  
  if( (i < max_x) && (j < max_y) ){
    
    int pixel_index = j*max_x + i;
    
    random = d_rand_state[pixel_index];
    
    Vector3 col(0,0,0);
    
    for(int s = 0; s < ns; s++){
    
      float u = float(i + curand_uniform(&random)) / float(max_x);
      float v = float(j + curand_uniform(&random)) / float(max_y);
      
      Ray r = (*cam)->get_ray(u,v);
      col += color(r, world, depth, &random);
      
    }
    
    d_rand_state[pixel_index] = random;
    
    col /= float(ns);
    
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    
    fb[pixel_index] = col;
    
  }
}



int main(int argc, char **argv)
{
  
  cudaEvent_t E0, E1, E2, E3, E4, E5;
  cudaEventCreate(&E0); cudaEventCreate(&E1);
  cudaEventCreate(&E2); cudaEventCreate(&E3);
  cudaEventCreate(&E4); cudaEventCreate(&E5);
  
  float createWorldTime, renderInitTime, renderTime, freeTime, totalTime;
  
  int nx, ny, ns, depth, dist, nthreads;
  
  parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads);

  int n = (2*dist)*(2*dist)+4;
  
  std::cerr << "Creating an image with (" << nx << "," << ny << ") pixels with " << nthreads << "x" << nthreads << std::endl;
  std::cerr << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cerr << "The world have " << n << " spheres." << std::endl;

  unsigned long long int seed = 1000;
  
  int num_pixels = nx*ny;
  float fb_size = num_pixels*sizeof(Vector3); // frame buffer, RGB para cada pixel
  
  Vector3 *frameBuffer;
  checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fb_size)); //Reservamos memoria

  
  dim3 blocks(nx/nthreads+1, ny/nthreads+1); // Dividmos el trabajo en tx x ty threads
  dim3 threads(nthreads,nthreads);
  
  Hitable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, n*sizeof(Hitable *)));
  Hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hitable *)));
  Camera **d_cam;
  checkCudaErrors(cudaMalloc((void **)&d_cam, sizeof(Camera *)));
  curandState *d_rand_state;
  size_t drand_size = num_pixels*sizeof(curandState);
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, drand_size));
  curandState *d_rand_state2;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));
  
  rand_init<<<1,1>>>(d_rand_state2, seed);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  create_world<<<1,1>>>(d_list, d_world, d_cam, nx, ny, dist, n, d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);
  
  cudaEventElapsedTime(&createWorldTime,E0,E1);
  
  std::cerr << "Create World Time: " << createWorldTime << " milisegs."<< std::endl;
  
  
  cudaEventRecord(E2,0);
  cudaEventSynchronize(E2);
  
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state, seed);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  cudaEventRecord(E3,0);
  cudaEventSynchronize(E3);
  
  cudaEventElapsedTime(&renderInitTime,E2,E3);
  
  std::cerr << "Render Init Time: " << renderInitTime << " milisegs."<< std::endl;
  
  cudaEventRecord(E4,0);
  cudaEventSynchronize(E4);
  
  render<<<blocks, threads>>>(frameBuffer, nx, ny, ns, d_cam, d_world, d_rand_state, depth);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  cudaEventRecord(E5,0);
  cudaEventSynchronize(E5);
  
  cudaEventElapsedTime(&renderTime,E4,E5);
  cudaEventElapsedTime(&totalTime,E0,E5);
  
  std::cerr << "Render Time: " << renderTime << " milisegs."<< std::endl;
  std::cerr << "Total Time: " << totalTime << " milisegs."<< std::endl;
  
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  
  for(int j = ny-1; j >= 0; j--){
    for(int i = 0; i < nx; i++){

      size_t pixel_index = j*nx + i;
      
      Vector3 col = frameBuffer[pixel_index];
      
      int ir = int(255.99*col.r());
      int ig = int(255.99*col.g());
      int ib = int(255.99*col.b());
      
      std::cout << ir << " " << ig << " " << ib << std::endl;
    }
  }
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world,d_cam, n);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_cam));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(frameBuffer));
  
  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);
  
  cudaEventElapsedTime(&freeTime, E0, E1);
  
  std::cerr << "Free Time: " << freeTime << " milisegs."<< std::endl;
  
  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2);
  cudaEventDestroy(E3); cudaEventDestroy(E4); cudaEventDestroy(E5);
  
  cudaDeviceReset();
  
  
  
}
