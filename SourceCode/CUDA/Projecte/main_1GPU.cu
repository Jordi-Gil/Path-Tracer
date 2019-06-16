#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#include "Sphere.cuh"
#include "HitableList.cuh"
#include "Camera.cuh"
#include "Material.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void error(const char *message){
  std::cout << message << std::endl;
  exit(0);
}

void help(){
  std::cout << "\n"  << std::endl;
  std::cout << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
  std::cout << "\t                size: (1200x600) | AAit: 10 | depth: 50 | spheres: 11 | nthreads: 8"  << std::endl;
  std::cout << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cout << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cout << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cout << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cout << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cout << "\t[-nthreads]     Number of threads to use" << std::endl;
  std::cout << "\t[-f][--file]    File name of pic generated." << std::endl;
  std::cout << "\t[-h][--help]    Show help." << std::endl;
  std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Examples of usage:" << std::endl;
  std::cout << "./path_tracing_1GPU -d"  << std::endl;
  std::cout << "./path_tracing_1GPU -nthreads 16 -sizeX 2000 -nthreads 32"<< std::endl;
  exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads, std::string &filename){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 512; ny = 512; ns = 50; depth = 50; dist = 11; nthreads = 32; filename = "pic.ppm";
  
  bool v_default = false;
  
  for(int i = 1; i < argc; i += 2){
    
    if(v_default) error("Error usage. Use [-h] [--help] to see the usage.");
    
  	if (std::string(argv[i]) == "-d" || std::string(argv[i]) == "--default") {
  		if ((i+1) < argc) error("The default parameter cannot have more arguments.");
  		std::cerr << "Default\n";
  		v_default = true;
  	} else if (std::string(argv[i]) == "-sizeX"){
  		if ((i+1) >= argc) error("-sizeX value expected");
  		nx = atoi(argv[i+1]);
  		if(nx == 0) error("-sizeX value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-sizeY"){
  		if ((i+1) >= argc) error("-sizeY value expected");
  		ny = atoi(argv[i+1]);
  		if(ny == 0) error("-sizeY value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-AAit"){
  		if ((i+1) >= argc) error("-AAit value expected");
  		ns = atoi(argv[i+1]);
  		if(ns == 0) error("-AAit value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-depth"){
  		if ((i+1) >= argc) error("-depth value expected");
  		depth = atoi(argv[i+1]);
  		if(depth == 0) error("-depth value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-spheres"){
  		if ((i+1) >= argc) error("-spheres value expected");
  		depth = atoi(argv[i+1]);
  		if(depth == 0) error("-spheres value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-nthreads"){
  		if ((i+1) >= argc) error("-nthreads value expected");
  		nthreads = atoi(argv[i+1]);
  		if(nthreads == 0) error("-nthreads value expected or cannot be 0");
  	} else if (std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
  		if ((i+1) >= argc) error("--file / -f value expected");
  		filename = std::string(argv[i+1]);
  		filename = filename+".ppm";
  	} else if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
  		help();
  	} else {
  		error("Error usage. Use [-h] [--help] to see the usage.");
  	}
  }
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
  if(result){
    std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << std::endl;
    std::cout << cudaGetErrorString(result) << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}

__global__ void free_world(Hitable **d_list, Hitable **d_world, Camera **d_cam) {
  int n = (*d_world)->length();
  for(int i = 0; i < n; i++){
    delete ((Sphere *)d_list[i])->mat_ptr;
    delete d_list[i];
  }
  delete *d_world;
  delete *d_cam;
}

#define Random (curand_uniform(&local_random))

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_cam, int nx, int ny, int dist, curandState *random){
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_random = *random;

    d_list[0] = new Sphere(Vector3(0,-1000,-1), 1000, new Lambertian(Vector3(0.5, 0.5, 0.5)));
    
    int i = 1;
    for (int a = -dist; a < dist; a++) {
      for (int b = -dist; b < dist; b++) {
        float material = Random; //defined in -> #define Random (curand_uniform(&local_random))
        Vector3 center(a+0.9*Random, 0.2, b+0.9*Random);

        if ((center-Vector3(0,0,0)).length() > 0.995) {
          if (material < 0.8) d_list[i++] = new Sphere(center, 0.2,new Lambertian(Vector3(Random*Random, Random*Random, Random*Random)));
          else if (material < 0.95) d_list[i++] = new Sphere(center, 0.2, new Metal(Vector3(0.5*(1.0+Random), 0.5*(1.0+Random), 0.5*(1.0+Random)),0.5*Random));
          else d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
        }
      }
    }

    d_list[i++] = new Sphere(Vector3( 0, 1, 0), 1.0, new Dielectric(1.5));
    d_list[i++] = new Sphere(Vector3(-4, 1, 0), 1.0, new Lambertian(Vector3(0.4, 0.2, 0.1)));
    d_list[i++] = new Sphere(Vector3( 4, 1, 0), 1.0, new Metal(Vector3(0.7, 0.6, 0.5),0.0));
    
    d_list[i++] = new Sphere(Vector3( 4, 1, 5), 1.0, new Metal(Vector3(0.9, 0.2, 0.2),0.0));

    *random = local_random;
    
    *d_world = new HitableList(d_list,i);
    
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
  std::string filename;
  
  parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, filename);

  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels with " << nthreads << "x" << nthreads << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << n << " spheres." << std::endl;

  unsigned long long int seed = 1000;
  
  int num_pixels = nx*ny;
  float fb_size = num_pixels*sizeof(Vector3); // frame buffer, RGB para cada pixel
  
  Vector3 *h_frameBuffer = (Vector3 *) malloc(fb_size);
  
  int nblocksX = (nx + nthreads - 1)/nthreads;
  int nblocksY = (ny + nthreads - 1)/nthreads;
  
  dim3 blocks(nblocksX, nblocksY, 1);
  dim3 threads(nthreads,nthreads, 1);
  
  /* Device variables */
  Vector3 *d_frameBuffer;
  Hitable **d_list;
  Hitable **d_world;
  Camera **d_cam;
  curandState *d_rand_state;
  curandState *d_rand_state2;
  
  size_t drand_size = num_pixels*sizeof(curandState);
  
  /* Allocate memory on device */
  cudaMallocManaged((void **)&d_frameBuffer, fb_size);
  cudaMalloc((void **)&d_list, n*sizeof(Hitable *));
  cudaMalloc((void **)&d_world, sizeof(Hitable *));
  cudaMalloc((void **)&d_cam, sizeof(Camera *));
  cudaMalloc((void **)&d_rand_state, drand_size);
  cudaMalloc((void **)&d_rand_state2, sizeof(curandState));
  
  
  rand_init<<<1,1>>>(d_rand_state2, seed);
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  create_world<<<1,1>>>(d_list, d_world, d_cam, nx, ny, dist, d_rand_state2);
  
  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);
  
  cudaEventElapsedTime(&createWorldTime,E0,E1);
  
  std::cout << "Create World Time: " << createWorldTime << " milisegs."<< std::endl;
  
  cudaEventRecord(E2,0);
  cudaEventSynchronize(E2);
  
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state, seed);
  
  cudaEventRecord(E3,0);
  cudaEventSynchronize(E3);
  
  cudaEventElapsedTime(&renderInitTime,E2,E3);
  
  std::cout << "Render Init Time: " << renderInitTime << " milisegs."<< std::endl;
  
  cudaEventRecord(E4,0);
  cudaEventSynchronize(E4);
  
  render<<<blocks, threads>>>(d_frameBuffer, nx, ny, ns, d_cam, d_world, d_rand_state, depth);
  
  cudaMemcpy(h_frameBuffer, d_frameBuffer, fb_size, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(E5,0);
  cudaEventSynchronize(E5);
  
  cudaEventElapsedTime(&renderTime,E4,E5);
  cudaEventElapsedTime(&totalTime,E0,E5);
  
  checkCudaErrors(cudaGetLastError());
  
  std::cout << "Render Time: " << renderTime << " milisegs."<< std::endl;
  std::cout << "Total Time: " << totalTime << " milisegs."<< std::endl;
  
  std::ofstream pic;
  pic.open(filename.c_str());
  
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
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  
  free_world<<<1,1>>>(d_list,d_world,d_cam);
  cudaFree(d_cam);
  cudaFree(d_world);
  cudaFree(d_list);
  cudaFree(d_rand_state);
  cudaFree(d_frameBuffer);
  
  cudaEventRecord(E1,0);
  cudaEventSynchronize(E1);
  
  cudaEventElapsedTime(&freeTime, E0, E1);
  
  std::cout << "Free Time: " << freeTime << " milisegs."<< std::endl;
  
  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2);
  cudaEventDestroy(E3); cudaEventDestroy(E4); cudaEventDestroy(E5);
  
  cudaDeviceReset();  
}
