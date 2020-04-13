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

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads, std::string &image, std::string &filename, int &numGPUs, bool &light, bool &random, bool &filter, int &diameter, float &gs, float &gr, bool &skybox, const int count){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; image = "random"; light = true; random = true;
	filter = false; gs = 0; gr = 0; diameter = 11; skybox = false;
  
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
    else if (std::string(argv[i]) == "-filter") {
      filter = true;
      diameter = atoi(argv[i+1]);
      i += 2;
      gs = atof(argv[i]);
      gr = atof(argv[i+1]);
    }
    else if(std::string(argv[i]) == "-skybox") {
      if((i+1) >= argc) error("-skybox value expected");
      if(std::string(argv[i+1]) == "ON") skybox = true;
      else if(std::string(argv[i+1]) == "OFF") skybox = false;
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

void properties(int numGPUs){
    
  std::cout << "GPU Info " << std::endl;

	for(int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		checkCudaErrors( cudaDeviceSetLimit( cudaLimitMallocHeapSize, 67108864 ) );
		checkCudaErrors( cudaDeviceSetLimit( cudaLimitStackSize, 131072 ) );
	}

	int device;
  cudaGetDevice(&device);
	cudaDeviceProp properties;
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

__device__ Vector3 color(const Ray& ray, Node *world, int depth, bool light, bool skybox, curandState *random, Skybox *sky){
  
  Ray cur_ray = ray;
  Vector3 cur_attenuation = Vector3::One();
  for(int i = 0; i < depth; i++){ 
    hit_record rec;
    if( world->checkCollision(cur_ray, 0.001, FLT_MAX, rec) ) {
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted(rec.u, rec.v);
      float pdf;
      if(rec.mat_ptr.scatter(cur_ray, rec, attenuation, scattered, pdf, random)){
        cur_attenuation *= attenuation;
        cur_attenuation += emitted;
        cur_ray = scattered;
      }
      else return cur_attenuation * emitted;
    }
    else {
      if(skybox && sky->hit(cur_ray, 0.00001, FLT_MAX, rec)){
        return cur_attenuation * rec.mat_ptr.emitted(rec.u, rec.v);
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

__device__ unsigned int findSplit(Triangle *d_list, int first, int last) {
    
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

__device__ int2 determineRange(Triangle *d_list, int idx, int objs) {
    
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

__global__ void setupCamera(Camera **d_cam, int nx, int ny, Camera cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_cam = new Camera(cam.getLookfrom(), cam.getLookat(), cam.getVUP(), cam.getFOV(), float(nx)/float(ny), cam.getAperture(), cam.getFocus(),0.0,0.1);
  }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state,unsigned long long seed, int minY, int maxY) {
  
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  
  int i = num%max_x;
  int j = num/max_x + minY;
  
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

__global__ void constructBVH(Node *d_internalNodes, Node *leafNodes, int objs, Triangle *d_list) {
    
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
  }
  
  if (split + 1 == last) {
    current->right = leafNodes + split + 1;
    (leafNodes + split + 1)->parent = current;
  }
  else{
    current->right = d_internalNodes + split + 1;
    (d_internalNodes + split + 1)->parent = current;
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

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Node *world, curandState *d_rand_state, int depth, bool light, bool skybox, Skybox *sky, int minY, int maxY) {

  int num = blockIdx.x*blockDim.x + threadIdx.x;

  int i = num%max_x;
  int j = num/max_x + minY;

  curandState local_random;

  int pixel_index = num;
    
  local_random = d_rand_state[pixel_index];
    
  Vector3 col(0,0,0);
    
  for(int s = 0; s < ns; s++){
    
    float u = float(i + cuRandom) / float(max_x);
    float v = float(j + cuRandom) / float(max_y);
      
    Ray r = (*cam)->get_ray(u, v, &local_random);
    col += color(r, world, depth, light, skybox, &local_random, sky);
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

  float totalTime;

  int nx, ny, ns, depth, dist, nthreads, numGPUs, diameter;
  bool light, random, filter, skybox;
  float gs,gr;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, image, filename, numGPUs, light, random, filter, diameter, gs, gr, skybox, 1);
  
  properties(numGPUs);

  /* Seed for CUDA cuRandom */
  unsigned long long int seed = 1000;

  /* #pixels of the image */
  int num_pixels = nx*ny;
  int elementsToJump = num_pixels/numGPUs;
	int bytesToJump = elementsToJump * sizeof(Vector3);
  int size = 0;

  /* Host variables */
  float fb_size = num_pixels*sizeof(Vector3);
  float drand_size = num_pixels*sizeof(curandState);
  float cam_size = sizeof(Camera*);
  Vector3 *h_frameBuffer;

  int blocks = (nx * ny)/(numGPUs * nthreads);

  /* Create world */
  Scene scene(dist, nx, ny);
  if(random) scene.loadScene(TRIANGL);
  else scene.loadScene(FFILE,filename);
	
	Camera cam = scene.getCamera();
	size = scene.getSize();
	
  float ob_size = size*sizeof(Triangle);
  float sky_size = sizeof(Skybox);
  int blocks2 = (size+nthreads-1)/(nthreads);
  
  std::cout << "Creating " << image << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << size << " objects." << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF" << std::endl;

  /* Device variables */
  Vector3 **d_frames = (Vector3 **) malloc(numGPUs * sizeof(Vector3));
  Triangle **d_objectsGPUs = (Triangle **) malloc(numGPUs * sizeof(Triangle));
  Camera ***d_cameras = (Camera ***) malloc(numGPUs * sizeof(Camera));
  curandState **d_randstates = (curandState **) malloc(numGPUs * sizeof(curandState));
  Node **d_internalNodes = (Node **) malloc(numGPUs * sizeof(Node));
  Node **d_leafNodes = (Node **) malloc(numGPUs * sizeof(Node));
  int **d_nodeCounters = (int **) malloc(numGPUs * sizeof(int));
  Skybox **d_skyboxes = (Skybox **) malloc(numGPUs * sizeof(Skybox));

  float internal_size = (size-1)*sizeof(Node);
  float leaves_size = size*sizeof(Node);
  
	cudaSetDevice(0);
  
  /* Allocate Memory Host */
  cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);

  /* Allocate memory on Device */
	
	cudaEvent_t E0, E1;
  cudaEventCreate(&E0); 
  cudaEventCreate(&E1);
  
  cudaEventRecord(E0,0);
  cudaEventSynchronize(E0);

  /* Allocate memory on Device */
  for(int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		
		Vector3 *d_frameBuffer;
		Triangle *d_objects;
		Camera **d_cam;
		curandState *d_rand_state;
		Node *d_internals;
		Node *d_leaves;
		int *d_nodeCounter;
		Skybox *d_skybox;
		
		cudaMallocManaged((void **)&d_frameBuffer, fb_size);
		cudaMalloc((void **)&d_objects, ob_size);
		cudaMalloc((void **)&d_cam, cam_size);
		cudaMalloc((void **)&d_rand_state, drand_size);
		cudaMalloc((void **)&d_internals, internal_size);
		cudaMalloc((void **)&d_leaves, leaves_size);
		cudaMalloc((void **)&d_nodeCounter, sizeof(int)*size);
		cudaMalloc((void **)&d_skybox, sizeof(Skybox));
		cudaMemset(d_nodeCounter, 0, sizeof(int)*size);
		cudaMemset(d_frameBuffer, 0, fb_size);
		
		d_frames[i] = d_frameBuffer;
		d_objectsGPUs[i] = d_objects;
		d_cameras[i] = d_cam;
		d_randstates[i] = d_rand_state;
		d_internalNodes[i] = d_internals;
		d_leafNodes[i] = d_leaves;
		d_nodeCounters[i] = d_nodeCounter;
		d_skyboxes[i] = d_skybox;
		
	}
	
	for(int i = 0; i < numGPUs; i++) {
	
		cudaSetDevice(i);
		
		Triangle *ob = scene.getObjects();
		Skybox *sky = scene.getSkybox();
	
		sky->hostToDevice(i);
    
		for(int j = 0; j < size; j++){
			ob[j].hostToDevice(i);
		}
		
		cudaMemcpy(d_objectsGPUs[i], ob, ob_size, cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
    
		cudaMemcpy(d_skyboxes[i], sky, sky_size, cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		
	}
  
	for(int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		
		setupCamera<<<1,1>>>(d_cameras[i],nx,ny, cam);
		checkCudaErrors(cudaGetLastError());

		render_init<<<blocks, nthreads>>>(nx, ny, d_randstates[i], seed, i*(ny/numGPUs), (i+1)*(ny/numGPUs));
		checkCudaErrors(cudaGetLastError());
		
		initLeafNodes<<<blocks2, nthreads>>>(d_leafNodes[i], size, d_objectsGPUs[i]);
		checkCudaErrors(cudaGetLastError());
		
		constructBVH<<<blocks2, nthreads>>>(d_internalNodes[i], d_leafNodes[i], size-1, d_objectsGPUs[i]);
		checkCudaErrors(cudaGetLastError());
	 
		boundingBoxBVH<<<blocks2, nthreads>>>(d_internalNodes[i], d_leafNodes[i], size, d_nodeCounters[i]);
		checkCudaErrors(cudaGetLastError());
		
		render<<<blocks, nthreads>>>(d_frames[i], nx, ny, ns, d_cameras[i], d_internalNodes[i], d_randstates[i], depth, light, skybox, d_skyboxes[i], i*(ny/numGPUs), (i+1)*(ny/numGPUs));
		checkCudaErrors(cudaGetLastError());
		
	}
	
  /* Copiamos del Device al Host*/
  
  for(int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		
		cudaMemcpyAsync(&h_frameBuffer[elementsToJump*i], d_frames[i], bytesToJump, cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		
  }
  
  for(int i = 0; i < numGPUs; i++){
		
		cudaSetDevice(i);
		cudaDeviceSynchronize();
		
	}
  
  cudaSetDevice(0);
  
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
  
  stbi_write_png(image.c_str(), nx, ny, 3, data, nx*3);
	
	if(filter){
    std::cout << "Filtering image using bilateral filter with Gs = " << gs << " and Gr = " << gr << " and window of diameter " << diameter << std::endl;
    std::string filenameFiltered = image.substr(0, image.length()-4) + "_Filtered.png";
    int sx, sy, sc;
    unsigned char *imageData = stbi_load(image.c_str(), &sx, &sy, &sc, 0);
    unsigned char *imageFiltered = new unsigned char[sx*sy*3];;
    bilateralFilter(diameter, sx, sy, imageData, imageFiltered, gs, gr);
    stbi_write_png(filenameFiltered.c_str(), sx, sy, 3, imageFiltered, sx*3);
  }
  
	
	for(int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		
		cudaFree(d_cameras[i]);
		cudaFree(d_objectsGPUs[i]);
		cudaFree(d_randstates[i]);
		cudaFree(d_frames[i]);
    cudaFree(d_skyboxes[i]);
    cudaFree(d_leafNodes[i]);
		cudaFree(d_internalNodes[i]);
		
	}
  
  cudaEventDestroy(E0);
  cudaEventDestroy(E1);
  
}