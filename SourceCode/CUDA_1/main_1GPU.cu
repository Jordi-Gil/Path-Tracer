#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#include "Camera.cuh"
#include "Scene.cuh"
#include "HitableList.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define Random (curand_uniform(&local_random))

void error(const char *message) {
  
  std::cout << message << std::endl;
  exit(0);
}

void format() {
  std::cout << "File format for scene." << std::endl;
  std::cout << "\t #          Comment, skip line." << std::endl;
  std::cout << "Spheres -> type center material" << std::endl;
  std::cout << "\t 1          Indicates that the 3D model is a Sphere object." << std::endl;
  std::cout << "\t Center     The center of the Sphere." << std::endl;
  std::cout << "\t Radius     The radius of the Sphere." << std::endl;
  std::cout << "\t Material -> type albedo [fuzz] [ref_idx]" << std::endl;
  std::cout << "\t\t 0        LAMBERTIAN" << std::endl;
  std::cout << "\t\t 1        METAL" << std::endl;
  std::cout << "\t\t 2        DIELECTRIC" << std::endl;
  std::cout << "\t\t 3        DIFFUSE LIGHT" << std::endl;
  std::cout << "\t\t albedo   Defines the color." << std::endl;
  std::cout << "\t\t fuzz     Only for METAL." << std::endl;
  std::cout << "\t\t ref_idx  Only for DIELECTRIC." << std::endl;
  std::cout << "Examples of declaration:\n" << std::endl;
  std::cout << "# my scene" << std::endl;
  std::cout << "Object   Center Rad Material  Albedo        Fuzz/ref_idx" << std::endl;
  std::cout << "1       0 1 0   2   1         0.5 0.78 0.9        " << std::endl;
  std::cout << "1       0 4 0   2   2         1   0    0.9    2   " << std::endl;
  std::cout << "1       1 4 1   2   3         0.9 0.9  0.9    1.5 " << std::endl;
}

void help(){

  std::cout << "\n"  << std::endl;
  std::cout << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
  std::cout << "\t                size: (2048x1080) | AAit: 10 | depth: 10 | spheres: 4 | nthreads: 32"  << std::endl;
  std::cout << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cout << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cout << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cout << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cout << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cout << "\t[-light]        Turn on/off the ambient light. Values can be ON/OFF" << std::endl;
  std::cout << "\t[-nthreads]     Number of threads to use" << std::endl;
  std::cout << "\t[-nGPUs]        Number of GPUs to distribute the work" << std::endl;
  std::cout << "\t[-i][--image]   File name of pic generated." << std::endl;
  std::cout << "\t[-f][--file]    File name of the scene." << std::endl;
  std::cout << "\t[-h][--help]    Show help." << std::endl;
  std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Examples of usage:" << std::endl;
  std::cout << "./path_tracing_NGPUs -d"  << std::endl;
  std::cout << "./path_tracing_NGPUs -nthreads 16 -sizeX 2000"<< std::endl;
  format();
  exit(1);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads, std::string &image, std::string &filename, int &numGPUs, bool &light, bool &random,const int count){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; image = "random"; light = true; random = true;
  
  nthreads = 32; numGPUs = 1;
  
  bool v_default = false;
  
  for(int i = 1; i < argc; i += 2){
    
    if(v_default) error("Error usage. Use [-h] [--help] to see the usage.");
    
    if (std::string(argv[i]) == "-d" || std::string(argv[i]) == "--default"){
      if((i+1) < argc) error("The default parameter cannot have more arguments.");
      std::cerr << "Default\n";
      v_default = true;
    }
    else if (std::string(argv[i]) == "-sizeX"){
      if((i+1) >= argc) error("-sizeX value expected");
      nx = atoi(argv[i+1]);
      if(nx == 0) error("-sizeX value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-sizeY"){
      if((i+1) >= argc) error("-sizeY value expected");
      ny = atoi(argv[i+1]);
      if(ny == 0) error("-sizeY value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-AAit"){
      if((i+1) >= argc) error("-AAit value expected");
      ns = atoi(argv[i+1]);
      if(ns == 0) error("-AAit value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-depth"){
      if((i+1) >= argc) error("-depth value expected");
      depth = atoi(argv[i+1]);
      if(depth == 0) error("-depth value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-spheres"){
      if((i+1) >= argc) error("-spheres value expected");
      dist = atoi(argv[i+1]);
      if(dist == 0) error("-spheres value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-nthreads"){
      if((i+1) >= argc) error("-nthreads value expected");
      nthreads = atoi(argv[i+1]);
      if(nthreads == 0) error("-nthreads value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-i" || std::string(argv[i]) == "--image"){
      if((i+1) >= argc) error("--image / -i file expected");
      filename = std::string(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
      if((i+1) >= argc) error("-name file expected");
      filename = std::string(argv[i+1]);
      image = filename;
      filename = filename+".txt";
      random = false;
    }
    else if(std::string(argv[i]) == "-nGPUs"){
      if((i+1) >= argc) error("-nGPUs value expected");
      numGPUs = atoi(argv[i+1]);
      if(numGPUs == 0) error("-nGPUs value expected or cannot be 0");
      numGPUs = std::min(numGPUs, count);
    }
    else if(std::string(argv[i]) == "-light") {
      if((i+1) >= argc) error("-light value expected");
      if(std::string(argv[i+1]) == "ON") light = true;
      else if(std::string(argv[i+1]) == "OFF") light = false;
    }
    else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
      help();
    }
    else{
      error("Error usage. Use [-h] [--help] to see the usage.");
    }
  }
  if(!light) image = image+"_noktem";
  image = image+".ppm";
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
  if(result){
    std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << std::endl;
    std::cout << cudaGetErrorString(result) << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}

void properties(){
    
  std::cout << "GPU Info " << std::endl;

  cudaSetDevice(0);
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp properties;
  checkCudaErrors( cudaDeviceSetLimit( cudaLimitMallocHeapSize, 67108864 ) );
  checkCudaErrors( cudaDeviceSetLimit( cudaLimitStackSize, 131072 ) );
  checkCudaErrors( cudaGetDeviceProperties( &properties, device ) );

  size_t limit1;
  checkCudaErrors( cudaDeviceGetLimit( &limit1, cudaLimitMallocHeapSize ) );
  size_t limit2;
  checkCudaErrors( cudaDeviceGetLimit( &limit2, cudaLimitStackSize ) );

  if( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) ) {
    std::cout << "Running on GPU " << device << " (" << properties.name << ")" << std::endl;
    std::cout << "Compute mode: " << properties.computeMode << std::endl;
    std::cout << "Concurrent Kernels: " << properties.concurrentKernels << std::endl;
    std::cout << "Warp size: " << properties.warpSize << std::endl;
    std::cout << "Major: " << properties.major << " Minor: " << properties.minor << std::endl;
    std::cout << "Cuda limit heap size: " << limit1 << std::endl;
    std::cout << "Cuda limit stack size: " << limit2 << "\n\n" << std::endl;
  }
  else std::cout << "GPU " << device << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
}

__device__ Vector3 color(const Ray& ray, HitableList **d_world, int depth, bool light, curandState *random){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3(1.0,1.0,1.0);
  for(int i = 0; i < depth; i++){ 
    hit_record rec;
    if( (*d_world)->checkCollision(cur_ray, 0.001, FLT_MAX, rec)) {
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted();
      if(rec.mat_ptr.scatter(cur_ray, rec, attenuation, scattered, random)){
        cur_attenuation *= attenuation;
        cur_attenuation += emitted;
        cur_ray = scattered;
      }
      else return cur_attenuation * emitted;
    }
    else {
      if(light) {
        Vector3 unit_direction = unit_vector(cur_ray.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        Vector3 c = (1.0 - t)*Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
        return cur_attenuation * c;
      }
      else return Vector3::Zero();
    }
  }
  return Vector3::Zero();
}

__global__ void setUpCameraWorld(Camera **d_cam, int nx, int ny, HitableList **d_world, Sphere *d_objects, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Vector3 lookfrom(13,2,3);
    Vector3 lookat(0,0,0);
    Vector3 up(0,1,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    *d_cam = new Camera(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus,0.0,0.1);
    
    *d_world = new HitableList(d_objects,size);
  }
}

__global__ void rand_init(curandState *random, int seed) {
  
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, 0, random);
  }

}

__global__ void render_init(int max_x, int max_y, curandState *rand_state,unsigned long long seed) {
  
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  
  int i = num%max_x;
  int j = num/max_x;
  
  if( (i >= max_x) || (j >= max_y) ) return;
    
  int pixel_index = num;
    
  curand_init((seed << 20) + pixel_index, 0, 0, &rand_state[pixel_index]);
  
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, HitableList **d_world, curandState *d_rand_state, int depth, bool light) {

  int num = blockIdx.x*blockDim.x + threadIdx.x;
 
  int i = num%max_x;
  int j = num/max_x;

  curandState local_random;

  int pixel_index = num;
    
  local_random = d_rand_state[pixel_index];
    
  Vector3 col(0,0,0);
    
  for(int s = 0; s < ns; s++){

    float u = float(i + Random) / float(max_x);
    float v = float(j + Random) / float(max_y);
      
    Ray r = (*cam)->get_ray(u, v, &local_random);
    col += color(r, d_world, depth, light, &local_random);
    
  }

  d_rand_state[pixel_index] = local_random;
    
  col /= float(ns);

  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);

  fb[pixel_index] = col;
    
}

int main(int argc, char **argv) {
  
  cudaDeviceReset();
  
  properties();

  cudaEvent_t E0, E1;
  cudaEventCreate(&E0); 
  cudaEventCreate(&E1);
  
  float totalTime;

  int nx, ny, ns, depth, dist, nthreads, numGPUs;
  bool light, random;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, image, filename, numGPUs, light, random, 1);

  /* Seed for CUDA Random */
  unsigned long long int seed = 1000;

  /* #pixels of the image */
  int num_pixels = nx*ny;
  int size = 0;

  /* Host variables */
  float fb_size = num_pixels*sizeof(Vector3);
  float drand_size = num_pixels*sizeof(curandState);
  float cam_size = sizeof(Camera*);
  float world_size = sizeof(HitableList*);
  Vector3 *h_frameBuffer;

  int blocks = (nx * ny)/(numGPUs * nthreads);

  /* Create world */
  Scene scene(dist);
  if(random) scene.loadScene(RANDOM);
  else scene.loadScene(FFILE,filename);
  
  size = scene.getSize();
  float ob_size = size*sizeof(Sphere);
  
  std::cout << "\nCreating " << image << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << size << " spheres." << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF" << std::endl;

  /* Device variables */
  Vector3 *d_frameBuffer;
  Sphere *d_objects;
  Camera **d_cam;
  HitableList **d_world;
  curandState *d_rand_state;
  
  /* Allocate Memory Host */
  cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);

  /* Allocate memory on device */
  cudaMallocManaged((void **)&d_frameBuffer, fb_size);
  cudaMalloc((void **)&d_objects, ob_size);
  cudaMalloc((void **)&d_world, world_size);
  cudaMalloc((void **)&d_cam, cam_size);
  cudaMalloc((void **)&d_rand_state, drand_size);
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  cudaMemcpy(d_objects, scene.getObjects(), ob_size, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaGetLastError());
  
  setUpCameraWorld<<<1,1>>>(d_cam, nx, ny, d_world, d_objects, size);
  checkCudaErrors( cudaGetLastError() );
  
  render_init<<<blocks, nthreads>>>(nx, ny, d_rand_state, seed);
  checkCudaErrors(cudaGetLastError());
  
  render<<<blocks, nthreads>>>(d_frameBuffer, nx, ny, ns, d_cam, d_world, d_rand_state, depth, light);
  checkCudaErrors(cudaGetLastError());

  cudaMemcpy(h_frameBuffer, d_frameBuffer, fb_size, cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaGetLastError());

  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);

  cudaEventElapsedTime(&totalTime,E0,E1);

  checkCudaErrors(cudaGetLastError());
  
  std::cout << "Total time: " << totalTime << " milisegs. " << std::endl;

  std::cout << "Generating file image..." << std::endl;
  std::ofstream pic;
  pic.open(image.c_str());

  pic << "P3\n" << nx << " " << ny << "\n255\n";
  
  for(int j = ny-1; j >= 0; j--){
    for(int i = 0; i < nx; i++){

      size_t pixel_index = j*nx + i;

      Vector3 col = h_frameBuffer[pixel_index];

      int ir = int(255.99*col.r());
      int ig = int(255.99*col.g());
      int ib = int(255.99*col.b());

      pic << ir << " " << ig << " " << ib << "\n";
    }
  }
  
  pic.close();

  cudaFree(d_cam);
  cudaFree(d_world);
  cudaFree(d_objects);
  cudaFree(d_rand_state);
  cudaFree(d_frameBuffer);

  cudaEventDestroy(E0); 
  cudaEventDestroy(E1);
}
