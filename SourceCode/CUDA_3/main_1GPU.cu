#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <limits>
#include <algorithm>
#include <stack>
#include <queue>

#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>


#include "Camera.cuh"
#include "Scene.cuh"
#include "Node.cuh"

//#define MAX 3.402823466e+38
//#define MIN 1.175494351e-38
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define cuRandom (curand_uniform(&local_random))

void print(Sphere *h_objects,int size){
  for(int i = 0; i < size; i++){
    std::cout << "Center(" << h_objects[i].center << "), Albedo(" << h_objects[i].mat_ptr.getAlbedo() << "), Material: " << h_objects[i].mat_ptr.getName() << std::endl;
  }
}

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
  std::cout << "\t                size: (1280x720) | AAit: 50 | depth: 50 | spheres: 11 | nthreads: 32"  << std::endl;
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
  exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads, std::string &image, std::string &filename, int &numGPUs, bool &light, bool &random, const int count){

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
      if(dist < 0) error("-spheres value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-nthreads"){
      if((i+1) >= argc) error("-nthreads value expected");
      nthreads = atoi(argv[i+1]);
      if(nthreads == 0) error("-nthreads value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-i" || std::string(argv[i]) == "--image"){
      if((i+1) >= argc) error("--image / -i value expected");
      image = std::string(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
      if((i+1) >= argc) error("--file / -f value expected");
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

__device__ Vector3 color(const Ray& ray, Node *world, int depth, bool light, curandState *random){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3::One();
  for(int i = 0; i < depth; i++){ 
    hit_record rec;
    if( world->checkCollision(cur_ray, 0.001, FLT_MAX, rec) ) {
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted();
      if(rec.mat_ptr.scatter(cur_ray, rec, attenuation, scattered, random)){
        cur_attenuation *= attenuation;
        cur_attenuation += emitted;
        cur_ray = scattered;
      }
      else return emitted;
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

__device__ unsigned int findSplit(Sphere *d_list, int first, int last) {
    
    long long firstCode = d_list[first].getMorton();
    long long lastCode = d_list[last].getMorton();
    
    if(firstCode == lastCode)
        return (first + last) >> 1;
     
    int commonPrefix = __clz(firstCode ^ lastCode);
    int split = first;
    
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        int newSplit = split + step; 
        
        if(newSplit < last){
      
            long long splitCode = d_list[newSplit].getMorton();
            
            int splitPrefix = __clz(firstCode ^ splitCode);
      
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
            }
        }
        
    } while (step > 1);
    
    return split;
        
}

__device__ int2 determineRange(Sphere *d_list, int idx, int objs) {
    
    int numberObj = objs-1;
    
    if(idx == 0)
        return make_int2(0,numberObj);
    
    long long idxCode = d_list[idx].getMorton();
    long long idxCodeUp = d_list[idx+1].getMorton();
    long long idxCodeDown = d_list[idx-1].getMorton();
    
    if((idxCode == idxCodeDown) and (idxCode == idxCodeUp)) {
    
        int idxInit = idx;
        bool dif = false;
        while(!dif and idx > 0 and idx < numberObj){
            ++idx;
            if(idx >= numberObj) dif = true;
                
            if(d_list[idx].getMorton() != d_list[idx+1].getMorton()) dif = true;
        }
        
        return make_int2(idxInit, idx);
        
    } else {
        
        int prefixUp = __clz(idxCode ^ idxCodeUp);
        int prefixDown = __clz(idxCode ^ idxCodeDown);
        
        int d = Helper::sgn( prefixUp - prefixDown );
        int dmin;
        
        if(d < 0) dmin = prefixUp;
        else if (d > 0) dmin = prefixDown;
        
        int lmax = 2;
        
        int newBoundary;
        int bitPrefix;
        do {
            
            newBoundary = idx + lmax * d;
            bitPrefix = -1;
            
            if(newBoundary >= 0 and newBoundary <= numberObj){
                long long newCode = d_list[idx + lmax * d].getMorton();
                
                bitPrefix = __clz(idxCode ^ newCode);
                if(bitPrefix > dmin) lmax *= 2;
                
            }
            
        } while(bitPrefix > dmin);
        
        int l = 0;
        
        for(int t = lmax/2; t >= 1; t /= 2){
            
            int newUpperBound = idx + (l + t) * d;
            
            if(newUpperBound <= numberObj and newUpperBound >= 0){
                long long splitCode = d_list[newUpperBound].getMorton();
                int splitPrefix = __clz(idxCode ^ splitCode);
                
                if(splitPrefix > dmin) l += t;
            }
            
        }
        
        int jdx = idx + l * d;
        
        if(jdx < idx) return make_int2(jdx,idx);
        else return make_int2(idx,jdx);
    }
}

__global__ void setupCamera(Camera **d_cam, int nx, int ny) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Vector3 lookfrom(13,2,3);
    Vector3 lookat(0,0,0);
    Vector3 up(0,1,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;
    *d_cam = new Camera(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus,0.0,0.1);
  }
}

__global__ void free_world(Sphere *d_objects, int size, Node *d_world, Camera **d_cam) {
  
  delete d_objects;
  delete d_world;
  delete *d_cam;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state,unsigned long long seed) {
  
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  
  int i = num%max_x;
  int j = num/max_x;
  
  if( (i >= max_x) || (j >= max_y) ) return;
    
  int pixel_index = num;
    
  curand_init((seed << 20) + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void initLeafNodes(Node *leafNodes, int objs, Sphere *d_list) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(idx >= objs) return;
  
  leafNodes[idx].obj = &d_list[idx];
  leafNodes[idx].box = d_list[idx].box;
  leafNodes[idx].id = idx;
}

__global__ void constructBVH(Node *d_internalNodes, Node *leafNodes, int objs, Sphere *d_list) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
  if(idx >= objs) return;

  int2 range = determineRange(d_list, idx, objs+1);
  
  int first = range.x;
  int last = range.y;
  
  int split = findSplit(d_list, first, last);
  
  Node *current = d_internalNodes + idx;
  
  if(split == first) {
    current->left = leafNodes + split;
    (leafNodes + split)->parent = current;
  }
  else{
    current->left = d_internalNodes + split;
    (d_internalNodes + split)->parent = current;
    (d_internalNodes + split)->id = split;
  }
  
  if (split + 1 == last) {
    current->right = leafNodes + split + 1;
    (leafNodes + split + 1)->parent = current;
  }
  else{
    current->right = d_internalNodes + split + 1;
    (d_internalNodes + split + 1)->parent = current;
    (d_internalNodes + split + 1)->id = split+1;
  }
    
}

__global__ void boundingBoxBVH(Node *d_internalNodes, Node *d_leafNodes, int objs, int *nodeCounter) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(idx >= objs) return;
  
  Node *leaf = &d_leafNodes[idx];
  
  Node* current = leaf->parent;
  
  int currentIdx = current - d_internalNodes;
  int res = atomicAdd(nodeCounter + currentIdx, 1);
    
  while (true) {
      
    if(res == 0){
        
      return;
    }

    aabb leftBoundingBox = current->left->box;
    aabb rightBoundingBox = current->right->box;

    current->box = surrounding_box(leftBoundingBox, rightBoundingBox);
    
    
    if (current == d_internalNodes) {
      return;
    }
    
    current = current->parent;
    currentIdx = current - d_internalNodes;
    res = atomicAdd(nodeCounter + currentIdx, 1);
      
  }
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Node *world, curandState *d_rand_state, int depth, bool light) {

  int num = blockIdx.x*blockDim.x + threadIdx.x;

  int i = num%max_x;
  int j = num/max_x;

  curandState local_random;

  int pixel_index = num;
    
  local_random = d_rand_state[pixel_index];
    
  Vector3 col(0,0,0);
    
  for(int s = 0; s < ns; s++){
    
    float u = float(i + cuRandom) / float(max_x);
    float v = float(j + cuRandom) / float(max_y);
      
    Ray r = (*cam)->get_ray(u, v, &local_random);
    col += color(r, world, depth, light, &local_random);
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
  checkCudaErrors(cudaGetLastError());

  float totalTime;

  int nx, ny, ns, depth, dist, nthreads, numGPUs;
  bool light, random;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, image, filename, numGPUs, light, random, 1);

  int n = (2*dist)*(2*dist)+5;

  std::cout << "Creating " << image << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << n << " spheres." << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF" << std::endl;

  /* Seed for CUDA cuRandom */
  unsigned long long int seed = 1000;

  /* #pixels of the image */
  int num_pixels = nx*ny;
  int size = 0;

  /* Host variables */
  float fb_size = num_pixels*sizeof(Vector3);
  float drand_size = num_pixels*sizeof(curandState);
  float cam_size = sizeof(Camera*);
  Vector3 *h_frameBuffer;
  Node *h_internalNodes;
  int *h_nodeCounter;

  int blocks = (nx * ny)/(numGPUs * nthreads);

  /* Create world */
  Scene scene(dist);
  if(random) scene.loadScene(RANDOM);
  else scene.loadScene(FFILE,filename);
  
  size = scene.getSize();
  std::cout << size << " esferas" << std::endl;
  float ob_size = size*sizeof(Sphere);

  int threads = nthreads;
  while(size < threads) threads /= 2;
  int blocks2 = (size+threads-1)/(numGPUs * threads);
  std::cout << "Threads: " << threads << std::endl;
  std::cout << "Block size: " << blocks2 << std::endl;

  /* Device variables */
  Vector3 *d_frameBuffer;
  Sphere *d_objects;
  Camera **d_cam;
  curandState *d_rand_state;
  Node *d_internalNodes;
  Node *d_leafNodes;
  int *d_nodeCounter;

  float internal_size = (size-1)*sizeof(Node);
  float leaves_size = size*sizeof(Node);
  
  /* Allocate Memory Host */
  cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);
  cudaMallocHost((Node **)&h_internalNodes, internal_size);
  cudaMallocHost((int **) &h_nodeCounter, sizeof(int)*size);
  checkCudaErrors(cudaGetLastError());

  /* Allocate memory on Device */
  cudaMallocManaged((void **)&d_frameBuffer, fb_size);
  cudaMalloc((void **)&d_objects, ob_size);
  cudaMalloc((void **)&d_cam, cam_size);
  cudaMalloc((void **)&d_rand_state, drand_size);
  cudaMalloc((void **)&d_internalNodes, internal_size);
  cudaMalloc((void **)&d_leafNodes, leaves_size);
  cudaMalloc((void **)&d_nodeCounter, sizeof(int)*size);
  cudaMemset(d_nodeCounter, 0, sizeof(int)*size);

  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);
  checkCudaErrors(cudaGetLastError());

  /* Copiamos del Host al Device */
  cudaMemcpy(d_objects, scene.getObjects(), ob_size, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaGetLastError());
  
  setupCamera<<<1,1>>>(d_cam,nx,ny);
  checkCudaErrors(cudaGetLastError());

  render_init<<<blocks, nthreads>>>(nx, ny, d_rand_state, seed);
  checkCudaErrors(cudaGetLastError());
  
  initLeafNodes<<<blocks2,threads>>>(d_leafNodes, size, d_objects);
  checkCudaErrors(cudaGetLastError());
  
  constructBVH<<<blocks2,threads>>>(d_internalNodes, d_leafNodes, size-1, d_objects);
  checkCudaErrors(cudaGetLastError());
  
  boundingBoxBVH<<<blocks2,threads>>>(d_internalNodes, d_leafNodes, size, d_nodeCounter);
  checkCudaErrors(cudaGetLastError());
  
  render<<<blocks, nthreads>>>(d_frameBuffer, nx, ny, ns, d_cam, d_internalNodes, d_rand_state, depth, light);
  checkCudaErrors(cudaGetLastError());

  /* Copiamos del Device al Host*/
  cudaMemcpy(h_frameBuffer, d_frameBuffer, fb_size, cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaGetLastError());
  
  cudaEventRecord(E1,0);
  checkCudaErrors(cudaGetLastError());

  cudaEventSynchronize(E1);
  checkCudaErrors(cudaGetLastError());

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
  cudaFree(d_objects);
  cudaFree(d_rand_state);
  cudaFree(d_frameBuffer);
  cudaFree(d_nodeCounter);
  cudaFree(d_leafNodes);
  cudaFree(d_internalNodes);
  cudaEventDestroy(E0);
  cudaEventDestroy(E1);
  
}
