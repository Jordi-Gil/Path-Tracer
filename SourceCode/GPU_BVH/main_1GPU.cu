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
#include "filters.hh"

#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define cuRandom (curand_uniform(&local_random))

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
  exit(1);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &image, std::string &filename, bool &light, bool &random, bool &filter, int &diameterBi, float &gs, float &gr, int &diameterMean, int &diameterMedian, bool &skybox, bool &oneTex, int &nthreads, int &numGPUs, const int count){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; image = "image";
  filter = false; gs = 0; gr = 0; diameterBi = 11; diameterMean = 3; diameterMedian = 3;
  
  skybox = false; oneTex = false;
  light = true; random = true;
  
  bool imageName = false;
  
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
    else if(std::string(argv[i]) == "-i" || std::string(argv[i]) == "--image"){
      if((i+1) >= argc) error("--image / -i file expected");
      filename = std::string(argv[i+1]);
      imageName = true;
    }
    else if(std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
      if((i+1) >= argc) error("-name file expected");
      filename = std::string(argv[i+1]);
      if(!imageName) image = filename;
      filename = filename+".txt";
      random = false;
    }
    else if(std::string(argv[i]) == "-light") {
      if((i+1) >= argc) error("-light value expected");
      if(std::string(argv[i+1]) == "ON") light = true;
      else if(std::string(argv[i+1]) == "OFF") light = false;
    }
    else if (std::string(argv[i]) == "-filter") {
      filter = true;
      diameterBi = atoi(argv[i+1]);
      
      i += 2;
      gs = atof(argv[i]);
      gr = atof(argv[i+1]);
      
      i+=2;
      diameterMean = atoi(argv[i]);
      diameterMedian = atoi(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-skybox") {
      if((i+1) >= argc) error("-skybox value expected");
      if(std::string(argv[i+1]) == "ON") skybox = true;
      else if(std::string(argv[i+1]) == "OFF") skybox = false;
    }
    else if(std::string(argv[i]) == "-oneTex") {
      if((i+1) >= argc) error("-oneTex value expected");
      if(std::string(argv[i+1]) == "ON") oneTex = true;
      else if(std::string(argv[i+1]) == "OFF") oneTex = false;
    }
    else if(std::string(argv[i]) == "-nGPUs"){
      if((i+1) >= argc) error("-nGPUs value expected");
      numGPUs = atoi(argv[i+1]);
      if(numGPUs == 0) error("-nGPUs value expected or cannot be 0");
      numGPUs = std::min(numGPUs, count);
    }
    else if(std::string(argv[i]) == "-nthreads"){
      if((i+1) >= argc) error("-nthreads value expected");
      nthreads = atoi(argv[i+1]);
      if(nthreads == 0) error("-nthreads value expected or cannot be 0");
    }
    else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
      help();
    }
    else{
      error("Error usage. Use [-h] [--help] to see the usage.");
    }
  }
  if(!light) image = image+"_noktem";
  image = image+".png";
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

__device__ Vector3 color(const Ray& ray, Node *world, int depth, bool light, bool skybox, curandState *random, Skybox *sky, bool oneTex, unsigned char **d_textures){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3::One();
  for(int i = 0; i < depth; i++){ 
    hit_record rec;
    if( world->intersect(cur_ray, 0.00001, FLT_MAX, rec) ) {
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted(rec.u, rec.v, oneTex, d_textures);
      
      if(rec.mat_ptr.scatter(cur_ray, rec, attenuation, scattered, random, oneTex, d_textures)){
        cur_attenuation *= attenuation;
        cur_attenuation += emitted;
        cur_ray = scattered;
      }
      else return cur_attenuation * emitted;
    }
    else {
      if(skybox && sky->hit(cur_ray, 0.00001, FLT_MAX, rec)){
        return cur_attenuation * rec.mat_ptr.emitted(rec.u, rec.v, oneTex, d_textures);
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
  }
  return Vector3::Zero();
}

__device__ int LongestCommonPrefix(int i, int j, int numObjects, Triangle *d_list) {
  
  if(i < 0 or i > numObjects - 1 or j < 0 or j > numObjects - 1) return -1;
  
  int codeI = d_list[i].getMorton();
  int codeJ = d_list[j].getMorton();
  
  if(i == j) {
    printf("Equals Longest\n");
    return __clz(codeI ^ codeJ);
  }
  else return __clz(codeI ^ codeJ);
  
}

__device__ int findSplit(Triangle *d_list, int first, int last) {
    
    
    if(first == last){
      return -1;
    }
    
    int firstCode = d_list[first].getMorton();
    int lastCode = d_list[last].getMorton();
    
    int commonPrefix = __clz(firstCode ^ lastCode);
    
    int split = first;
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        int newSplit = split + step; 
        
        if(newSplit < last){
      
            int splitCode = d_list[newSplit].getMorton();
            
            int splitPrefix = __clz(firstCode ^ splitCode);
      
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
            }
        }
        
    } while (step > 1);
    
    return split;
        
}

__device__ int2 determineRange(Triangle *d_list, int idx, int objs) {

  
    int d =  LongestCommonPrefix(idx, idx + 1, objs, d_list) - 
             LongestCommonPrefix(idx, idx - 1, objs, d_list) >= 0 ? 1 : -1;
    
    int dmin = LongestCommonPrefix(idx, idx - d, objs, d_list);
    
    int lmax = 2;
    
    while(LongestCommonPrefix(idx, idx + lmax*d, objs, d_list) > dmin){
      lmax <<=1;
    }
    
    int l = 0;
    int div = 2;
    
    for(int t = lmax/div; t >= 1; t >>= 1) {
      
      if(LongestCommonPrefix(idx, idx + (l + t) * d, objs, d_list) > dmin) l += t;
        
    }
    
    int jdx = idx + l * d;
        
    if(jdx < idx) return make_int2(jdx,idx);
    else return make_int2(idx,jdx);
    
}

__global__ void setupCamera(Camera **d_cam, int nx, int ny, Camera cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_cam = new Camera(cam.getLookfrom(), cam.getLookat(), cam.getVUP(), cam.getFOV(), float(nx)/float(ny), cam.getAperture(), cam.getFocus(),0.0,0.1);
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

__global__ void initLeafNodes(Node *leafNodes, int objs, Triangle *d_list) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(idx >= objs) return;
  
  leafNodes[idx].obj = &d_list[idx];
  leafNodes[idx].box = d_list[idx].getBox();

}

__global__ void boundingBoxBVH(Node *d_internalNodes, Node *d_leafNodes, int objs, int *nodeCounter) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(idx >= objs) return;
  
  Node *leaf = d_leafNodes + idx;
  
  Node* current = leaf->parent;
  int currentIdx = current - d_internalNodes;
  int res = atomicAdd(nodeCounter + currentIdx, 1);
    
  while (true) {
      
    if(res == 0) return;

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

__global__ void constructBVH(Node *d_internalNodes, Node *leafNodes, int objs, Triangle *d_list) {
    
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
  if(idx >= objs) return;
  
  int2 range = determineRange(d_list, idx, objs+1);
  
  int first = range.x;
  int last = range.y;
  
  int split = findSplit(d_list, first, last);
  
  if(split == -1){
    split = (first+last) >> 1;
    ++last;
  }
  
  Node *current = d_internalNodes + idx;
  
  if(split == first) {
    current->left = leafNodes + split;
    current->left->isLeaf = true;
    current->left->isLeft = true;
    (leafNodes + split)->parent = current;
  }
  else{
    current->left = d_internalNodes + split;
    current->left->isLeft = true;
    (d_internalNodes + split)->parent = current;
  }
  
  if (split + 1 == last) {
    current->right = leafNodes + split + 1;
    current->right->isLeaf = true;
    current->right->isRight = true;
    (leafNodes + split + 1)->parent = current;
  }
  else{
    current->right = d_internalNodes + split + 1;
    current->right->isRight = true;
    (d_internalNodes + split + 1)->parent = current;
  }
    
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Node *world, curandState *d_rand_state, int depth, bool light, bool skybox, Skybox *sky, bool oneTex, unsigned char ** d_textures) {

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
    col += color(r, world, depth, light, skybox, &local_random, sky, oneTex, d_textures);
  }
    
  d_rand_state[pixel_index] = local_random;

  col /= float(ns);

  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);

  fb[pixel_index] = col;
}

__global__ void checkBVH(Node *d_internalNodes, Node *d_leaves, int objs){
  
  if (threadIdx.x == 0 && blockIdx.x == 0){
    
    printf("Checking BVH...\n");
    
    for(int i = 0; i < objs; i++){
      
      if(!d_leaves[i].parent){
        printf("Leaf without parent %d\n",i);
      }
    }
    
    for(int i = 0; i < objs-1; i++){
      
      if(!d_internalNodes[i].left){
        printf("Internal without left %d\n",i);
      }
      
      if(!d_internalNodes[i].right){
        printf("Internal without right %d\n",i);
      }
      
      if(!d_internalNodes[i].parent){
        printf("Internal without parent %d\n",i);
      }
      
    }
    printf("BVH checked!\n");
  }
}

int main(int argc, char **argv) {
    
  cudaDeviceReset();

  properties();

  cudaEvent_t E0, E1;
  cudaEventCreate(&E0); 
  cudaEventCreate(&E1);

  float totalTime;

  int nx, ny, ns, depth, dist, diameterBi, diameterMean, diameterMedian, nthreads, numGPUs;
  bool light, random, filter, skybox, oneTex;
  float gs, gr;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, image, filename, light, random, filter, diameterBi, gs, gr, diameterMean, diameterMedian, skybox, oneTex, nthreads, numGPUs, 1);

  /* Seed for CUDA cuRandom */
  unsigned long long int seed = 1000;

  /* #pixels of the image */
  int num_pixels = nx*ny;
  int size = 0;
  int num_textures = 0;

  /* Host variables */
  float fb_size = num_pixels*sizeof(Vector3);
  float drand_size = num_pixels*sizeof(curandState);
  float cam_size = sizeof(Camera*);
  Vector3 *h_frameBuffer;

  int blocks = (nx * ny)/(numGPUs * nthreads);

  /* Create world */
  Scene scene(dist, nx, ny);
  if(random) scene.loadScene(TRIANGL);
  else scene.loadScene(FFILE,filename,oneTex);
	
	Triangle *h_objects = scene.getObjects();
  Skybox *h_skybox = scene.getSkybox();
  unsigned char **textures; 
  unsigned char **h_textures;
  Vector3 *textureSizes;
  if(oneTex){
    textures = scene.getTextures();
    textureSizes = scene.getTextureSizes();
    num_textures = scene.getNumTextures();
  }
  
  size = scene.getSize();
  float ob_size = size*sizeof(Triangle);

  int threads = nthreads;
  while(size < threads) threads /= 2;
  int blocks2 = (size+threads-1)/(numGPUs * threads);
  
  std::cout << "Creating " << image << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << size << " objects." << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF" << std::endl;

  /* Device variables */
  Vector3 *d_frameBuffer;
  Triangle *d_objects;
  Camera **d_cam;
  curandState *d_rand_state;
  Node *d_internalNodes;
  Node *d_leafNodes;
  int *d_nodeCounter;
  Skybox *d_skybox;
  unsigned char **d_textures;

  float internal_size = (size-1)*sizeof(Node);
  float leaves_size = size*sizeof(Node);
  
  /* Allocate Memory Host */
  cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);

  /* Allocate memory on Device */
  cudaMallocManaged((void **)&d_frameBuffer, fb_size);
  cudaMalloc((void **)&d_objects, ob_size);
  cudaMalloc((void **)&d_cam, cam_size);
  cudaMalloc((void **)&d_rand_state, drand_size);
  cudaMalloc((void **)&d_internalNodes, internal_size);
  cudaMalloc((void **)&d_leafNodes, leaves_size);
  cudaMalloc((void **)&d_nodeCounter, sizeof(int)*size - 1);
  cudaMemset(d_nodeCounter, 0, sizeof(int)*size - 1);
  cudaMalloc((void **)&d_skybox, sizeof(Skybox));

  if(num_textures > 0){
    int count = 0;
    for(int i = 0; i < num_textures; i++){
      Vector3 p = textureSizes[i];
      count += (p[0]*p[1]*p[2]);
    }
    
    h_textures = (unsigned char **) malloc(sizeof(unsigned char)*count);
    
    std::cout << "Binding textures" << std::endl;
    for(int i = 0; i < num_textures; i++){
      std::cout << "Texture " << i << std::endl;
      
      Vector3 p = textureSizes[i];
      unsigned char *image = textures[i];
        
      cudaMalloc((void**)&h_textures[i], sizeof(unsigned char)*p[0]*p[1]*p[2]);
      cudaMemcpy(h_textures[i], image, sizeof(unsigned char)*p[0]*p[1]*p[2], cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&d_textures, sizeof(unsigned char *) * num_textures);
    cudaMemcpy(d_textures, h_textures, sizeof(unsigned char*) * num_textures, cudaMemcpyHostToDevice);
  }

  if(!oneTex){
    for(int i = 0; i < size; i++){
      h_objects[i].hostToDevice(0);
    }
  }
  
  h_skybox->hostToDevice(0);
  
  std::cout << "Start" << std::endl;
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);

  cudaMemcpy(d_skybox, h_skybox, sizeof(Skybox), cudaMemcpyHostToDevice);
  checkCudaErrors(cudaGetLastError());
  
  cudaMemcpy(d_objects, h_objects, ob_size, cudaMemcpyHostToDevice);
  checkCudaErrors(cudaGetLastError());
  
  setupCamera<<<1,1>>>(d_cam,nx,ny, scene.getCamera());
  checkCudaErrors(cudaGetLastError());

  render_init<<<blocks, nthreads>>>(nx, ny, d_rand_state, seed);
  checkCudaErrors(cudaGetLastError());
  
  initLeafNodes<<<blocks2,threads>>>(d_leafNodes, size, d_objects);
  checkCudaErrors(cudaGetLastError());
  
  std::cout << "constructBVH..." << std::endl;
  constructBVH<<<blocks2,threads>>>(d_internalNodes, d_leafNodes, size-1, d_objects);
  checkCudaErrors(cudaGetLastError());
  
//   checkBVH<<<1,1>>>(d_internalNodes, d_leafNodes, size);
//   checkCudaErrors(cudaGetLastError());
  
  std::cout << "boundingBoxBVH..." << std::endl;
  boundingBoxBVH<<<blocks2,threads>>>(d_internalNodes, d_leafNodes, size, d_nodeCounter);
  checkCudaErrors(cudaGetLastError());
  
  std::cout << "Rendering..." << std::endl;
  render<<<blocks, nthreads>>>(d_frameBuffer, nx, ny, ns, d_cam, d_internalNodes, d_rand_state, depth, light, skybox, d_skybox, oneTex, d_textures);
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
  uint8_t *data = new uint8_t[nx*ny*3];
  int count = 0;
  for(int j = ny-1; j >= 0; j--){
    for(int i = 0; i < nx; i++){

      size_t pixel_index = j*nx + i;
      
      Vector3 col = h_frameBuffer[pixel_index];
      
      int ir = int(255.99*col.r());
      int ig = int(255.99*col.g());
      int ib = int(255.99*col.b());

      data[count++] = ir;
      data[count++] = ig;
      data[count++] = ib;
    }
  }
  
  cudaFree(d_cam);
  cudaFree(d_objects);
  cudaFree(d_rand_state);
  cudaFree(d_frameBuffer);
  cudaFree(d_nodeCounter);
  cudaFree(d_leafNodes);
  cudaFree(d_internalNodes);
  
  cudaEventDestroy(E0);
  cudaEventDestroy(E1);
  
  image = "../Resources/Images/GPU_BVH/"+image;
  
  stbi_write_png(image.c_str(), nx, ny, 3, data, nx*3);

  if(filter){
    std::cout << "Filtering image using bilateral filter with Gs = " << gs << " and Gr = " << gr << " and window of diameter " << diameterBi << std::endl;
    std::string filenameFiltered = image.substr(0, image.length()-4) + "_bilateral_filter.png";
    int sx, sy, sc;
    unsigned char *imageData = stbi_load(image.c_str(), &sx, &sy, &sc, 0);
    unsigned char *imageFiltered = new unsigned char[sx*sy*3];
    
    bilateralFilter(diameterBi, sx, sy, imageData, imageFiltered, gs, gr);
    stbi_write_png(filenameFiltered.c_str(), sx, sy, 3, imageFiltered, sx*3);
    
    std::cout << "Filtering image using median filter  with window of diameter " << diameterMedian << std::endl;
    filenameFiltered = image.substr(0, image.length()-4) + "_median_filter.png";
    
    medianFilter(diameterMedian, sx, sy, imageData, imageFiltered);
    stbi_write_png(filenameFiltered.c_str(), sx, sy, 3, imageFiltered, sx*3);
    
    std::cout << "Filtering image using mean filter with window of diameter " << diameterMean << std::endl;
    filenameFiltered = image.substr(0, image.length()-4) + "_mean_filter.png";
    
    meanFilter(diameterMean,sx, sy, imageData, imageFiltered);
    stbi_write_png(filenameFiltered.c_str(), sx, sy, 3, imageFiltered, sx*3);
  }
}
