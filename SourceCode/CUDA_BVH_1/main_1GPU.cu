#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <limits>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <stack>
#include <queue>

#include "MovingSphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Node.cuh"

#define MAX 3.402823466e+38
#define MIN 1.175494351e-38
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define cuRandom (curand_uniform(&local_random))
#define Random (rand()/(RAND_MAX + 1.0))

void error(const char *message) {
  
    std::cout << message << std::endl;
    exit(0);
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
	std::cout << "\t[-nthreads]     Number of threads to use" << std::endl;
	std::cout << "\t[-nGPUs]        Number of GPUs to distribute the work" << std::endl;
	std::cout << "\t[-f][--file]    File name of pic generated." << std::endl;
	std::cout << "\t[-h][--help]    Show help." << std::endl;
	std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
	std::cout << "\n" << std::endl;
	std::cout << "Examples of usage:" << std::endl;
	std::cout << "./path_tracing_NGPUs -d"  << std::endl;
	std::cout << "./path_tracing_NGPUs -nthreads 16 -sizeX 2000"<< std::endl;
	exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, int &nthreads, std::string &filename, int &numGPUs,const int count){
  
	if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
	
	nx = 720; ny = 348; ns = 50; depth = 50; dist = 11; nthreads = 32; filename = "pic.ppm"; numGPUs = 1;
  
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
		else if(std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
			if((i+1) >= argc) error("-name file expected");
			filename = std::string(argv[i+1]);
			filename = filename+".ppm";
		}
		else if(std::string(argv[i]) == "-nGPUs"){
			if((i+1) >= argc) error("-nGPUs value expected");
			numGPUs = atoi(argv[i+1]);
			if(numGPUs == 0) error("-nGPUs value expected or cannot be 0");
			numGPUs = std::min(numGPUs, count);
		}
		else if(std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
			help();
		}
		else{
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
    
	if( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) )
	{
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

void iterativeTraversal(Node* root) {
    
    std::stack <Node *> stack;
    std::queue <Node *> queue;
    
    std::cout << "Root" << std::endl;
    queue.push(root);
    int i = 0;
    while(!queue.empty()) {
        std::cout << i << std::endl;
        
        root = queue.front();

        queue.pop();
        stack.push(root);
        
        if(root->left){
            std::cout << "Left" << std::endl;
            queue.push(root->left);
        }
        if(root->right){
            std::cout << "Right" << std::endl;
            queue.push(root->right);
        }
        i++;
    }
    
    while(!stack.empty()) {
        
        root = stack.top();
        
        if(root->obj) {
            root->box = root->obj->getBox(); //Leaf node
        }
        else { //Internal node
            
            aabb left_aabb = root->left->box;
            aabb right_aabb = root->right->box;
            
            root->box = surrounding_box(left_aabb, right_aabb);
            
        }
        stack.pop();
    }
}

void compare(Vector3 &max, Vector3 &min, Vector3 point) {
    
    if(point[0] > max[0]) max[0] = point[0]; //x
    if(point[1] > max[1]) max[1] = point[1]; //y
    if(point[2] > max[2]) max[2] = point[2]; //z
    
    if(point[0] < min[0]) min[0] = point[0]; //x
    if(point[1] < min[1]) min[1] = point[1]; //y
    if(point[2] < min[2]) min[2] = point[2]; //z
}

void create_world(MovingSphere *h_objects, Camera **h_cam, int &size, int nx, int ny, int dist){

    Vector3 max(MIN);
    Vector3 min(MAX);
    
    int i = 0;
        
    h_objects[i] = MovingSphere(Vector3(0,-1000,-1), Vector3(0,-1000,-1), 0.0, 1.0, 1000, new Lambertian(Vector3(0.5, 0.5, 0.5))); 
    
    compare(max,min, h_objects[i].getCenter());
    i++;
    for (int a = -dist; a < dist; a++) {
        for (int b = -dist; b < dist; b++) {
            float material = Random;
            Vector3 center(a+0.9*Random, 0.2, b+0.9*Random);

            if ((center-Vector3(0,0,0)).length() > 0.995) {
                if (material < 0.8) h_objects[i] = MovingSphere(center, center+Vector3(0,0.5*Random,0),0.0,1.0,.2,new Lambertian(Vector3(Random*Random, Random*Random, Random*Random)));
                else if (material < 0.95) h_objects[i] = MovingSphere(center, center, 0.0, 1.0, 0.2, new Metal(Vector3(0.5*(1.0+Random), 0.5*(1.0+Random), 0.5*(1.0+Random)),0.5*Random));
                else h_objects[i] = MovingSphere(center, center, 0.0, 1.0, 0.2, new Dielectric(1.5));
                
                compare(max,min,h_objects[i].getCenter());
                i++;
                
            }
        }
    }
    
    h_objects[i] = MovingSphere(Vector3( 0, 1, 0), Vector3( 0, 1, 0), 0.0, 1.0, 1.0, new Dielectric(1.5));
    compare(max,min,h_objects[i].getCenter()); i++;
    h_objects[i] = MovingSphere(Vector3(-4, 1, 0), Vector3(-4, 1, 0), 0.0, 1.0, 1.0, new Lambertian(Vector3(0.4, 0.2, 0.1)));
    compare(max,min,h_objects[i].getCenter()); i++;
    h_objects[i] = MovingSphere(Vector3( 4, 1, 0), Vector3( 4, 1, 0), 0.0, 1.0, 1.0, new Metal(Vector3(0.7, 0.6, 0.5),0.0));
    compare(max,min,h_objects[i].getCenter()); i++;
    h_objects[i] = MovingSphere(Vector3( 4, 1, 5), Vector3( 4, 1, 5), 0.0, 1.0, 1.0, new Metal(Vector3(0.9, 0.2, 0.2),0.0));
    compare(max,min,h_objects[i].getCenter()); i++;
    
    float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
    float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
    
    size = i;
    
    for(int idx = 0; idx < size; idx++){
        Vector3 point = h_objects[idx].getCenter();
            
        point[0] = ((point[0] - min[0])/(max[0] - min[0]));
        point[1] = ((point[1] - min[1])/(max[1] - min[1]));
        point[2] = ((point[2] - min[2])/(max[2] - min[2]));
        
        h_objects[idx].setMorton(Helper::morton3D(point[0],point[1],point[2])+idx);
    }
    
    std::sort(h_objects, h_objects + size , ObjEval());
    
    Vector3 lookfrom(13,2,3);
    Vector3 lookat(0,0,0);
    Vector3 up(0,1,0);
    float dist_to_focus = 10;
    float aperture = 0.1;
    *h_cam = new Camera(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 0.1);
    
}

__device__ Vector3 color(const Ray& ray, Node *world, int depth, curandState *random){
  
    Ray cur_ray = ray;
    Vector3 cur_attenuation = Vector3(1.0,1.0,1.0);
	for(int i = 0; i < depth; i++){ 
		hit_record rec;
		if( world->checkCollision(cur_ray, 0.001, FLT_MAX, rec)) {
      
			Ray scattered;
			Vector3 attenuation;
			if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, random)){
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else return Vector3(0.0,0.0,0.0);
		}
		else {
			Vector3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5*(unit_direction.y() + 1.0);
			Vector3 c = (1.0 - t)*Vector3(1.0,1.0,1.0) + t*Vector3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	
	return Vector3(0.0,0.0,0.0);
	
}

__device__ unsigned int findSplit(MovingSphere *d_list, int first, int last) {
    
    unsigned int firstCode = d_list[first].getMorton();
    unsigned int lastCode = d_list[last].getMorton();
    
    if(firstCode == lastCode)
        return (first + last) >> 1;
     
    int commonPrefix = __clz(firstCode ^ lastCode);
    int split = first;
    
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        int newSplit = split + step; 
        
        if(newSplit < last){
      
            unsigned int splitCode = d_list[newSplit].getMorton();
            
            int splitPrefix = __clz(firstCode ^ splitCode);
      
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
            }
        }
        
    } while (step > 1);
    
    return split;
        
}

__device__ int2 determineRange(MovingSphere *d_list, int idx, int objs) {
    
    //printf("%d\n",idx);
    
    int numberObj = objs-1;
    
    if(idx == 0)
        return make_int2(0,numberObj);
    
    unsigned int idxCode = d_list[idx].getMorton();
    //printf("CodeIdx: %d \n",idxCode);
    unsigned int idxCodeUp = d_list[idx+1].getMorton();
    unsigned int idxCodeDown = d_list[idx-1].getMorton();
    
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
                unsigned int newCode = d_list[idx + lmax * d].getMorton();
                
                bitPrefix = __clz(idxCode ^ newCode);
                if(bitPrefix > dmin) lmax *= 2;
                
            }
            
        } while(bitPrefix > dmin);
        
        int l = 0;
        
        for(int t = lmax/2; t >= 1; t /= 2){
            
            int newUpperBound = idx + (l + t) * d;
            
            if(newUpperBound <= numberObj and newUpperBound >= 0){
                unsigned int splitCode = d_list[newUpperBound].getMorton();
                int splitPrefix = __clz(idxCode ^ splitCode);
                
                if(splitPrefix > dmin) l += t;
            }
            
        }
        
        int jdx = idx + l * d;
        
        if(jdx < idx) return make_int2(jdx,idx);
        else return make_int2(idx,jdx);
    }
}

__global__ void free_world(MovingSphere *d_objects, int size, Hitable **d_world, Camera **d_cam) {

	for(int i = 0; i < size; i++){
		delete d_objects[i].mat_ptr;
	}
	delete *d_world;
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

__global__ void initLeafNodes(Node *leafNodes, int objs, MovingSphere *d_list) {
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx >= objs) return;
    
    leafNodes[idx].obj = &d_list[idx];
}

__global__ void constructBVH(Node *d_internalNodes, Node *leafNodes, int objs, MovingSphere *d_list) {
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(idx >= objs) return;

    int2 range = determineRange(d_list, idx, objs+1);
    
    int first = range.x;
    int last = range.y;
    
    int split = findSplit(d_list, first, last);
    
    Node *current = d_internalNodes + idx;
    
    if(split == first) {
        current->left = leafNodes + split;
        (leafNodes+split)->parent = current;
    }
    else{ 
        current->left = d_internalNodes + split;
        (d_internalNodes + split)->parent = current;
    }
    
    if (split + 1 == last) {
        current->left = leafNodes + split + 1;
        (leafNodes + split + 1)->parent = current;
    }
    else{
        current->left = d_internalNodes + split + 1;
        (d_internalNodes + split + 1)->parent = current;
    }
    
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Node *world, curandState *d_rand_state, int depth) {
	
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
		col += color(r, world, depth, &local_random);
      
	}
    
	d_rand_state[pixel_index] = local_random;
    
	col /= float(ns);
    
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    
    fb[pixel_index] = col;
    
}

int main(int argc, char **argv) {
	
    properties();

	cudaEvent_t E0, E1;
	cudaEventCreate(&E0); 
    cudaEventCreate(&E1);
    checkCudaErrors(cudaGetLastError());
  
	float totalTime;
  
	int nx, ny, ns, depth, dist, nthreads, numGPUs;
	std::string filename;
  
	parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, filename, numGPUs, 1);

	int n = (2*dist)*(2*dist)+5;
	
	std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
	std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
	std::cout << "The world have " << n << " spheres." << std::endl;

	/* Seed for CUDA cuRandom */
	unsigned long long int seed = 1000;
  
	/* #pixels of the image */
	int num_pixels = nx*ny;
    int size = 0;
	
	/* Host variables */
	float fb_size = num_pixels*sizeof(Vector3);
    float ob_size = n*sizeof(MovingSphere);
	float drand_size = num_pixels*sizeof(curandState);
    float cam_size = sizeof(Camera*);
	Vector3 *h_frameBuffer;
    MovingSphere *h_objects;
    Camera **h_cam;
    Node *h_internalNodes;
    MovingSphere *h_objects_aux;
    
	int blocks = (nx * ny)/(numGPUs * nthreads);
	
	/* Allocate Memory Host */
	cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);
    cudaMallocHost((MovingSphere**)&h_objects, ob_size);
    cudaMallocHost((MovingSphere**)&h_objects_aux, ob_size);
    cudaMallocHost((Camera **)&h_cam, cam_size);
    
    /* Device variables */
    Vector3 *d_frameBuffer;
    MovingSphere *d_objects;
    Camera **d_cam;
    curandState *d_rand_state;
    Node *d_internalNodes;
    Node *leafNodes;
  
    /* Create world */
    std::cout << "Creating world..." << std::endl;
    create_world(h_objects, h_cam, size, nx, ny, dist);
    std::cout << "Wolrd created" << std::endl;
    
    std::cout << size << " esferas" << std::endl;
    ob_size = size*sizeof(MovingSphere);
    int threads = nthreads;
    while(size < threads) threads /= 2;
    int blocks2 = size/threads;
    std::cout << "Threads: " << threads << std::endl;
    
    float internal_size = (size-1)*sizeof(Node);
    
    cudaMallocHost((Node **)&h_internalNodes, internal_size);
    checkCudaErrors(cudaGetLastError());
    
    /* Allocate memory on Device */
    cudaMallocManaged((void **)&d_frameBuffer, fb_size);
    cudaMalloc((void **)&d_objects, ob_size);
    cudaMalloc((void **)&d_cam, cam_size);
    cudaMalloc((void **)&d_rand_state, drand_size);
    cudaMalloc((void **)&d_internalNodes, internal_size);
    cudaMalloc((void **)&leafNodes, size*sizeof(Node));
    
    cudaEventRecord(E0,0);
    cudaEventSynchronize(E0);
    checkCudaErrors(cudaGetLastError());
    
    /* Copiamos del Host al Device */
    cudaMemcpy(d_objects, h_objects, ob_size, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(d_cam, h_cam, cam_size, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    
    render_init<<<blocks, nthreads>>>(nx, ny, d_rand_state, seed);
    checkCudaErrors(cudaGetLastError());
    
    initLeafNodes<<<blocks2,threads>>>(leafNodes, (size-1), d_objects);
    checkCudaErrors(cudaGetLastError());
    
    constructBVH<<<blocks2,threads>>>(d_internalNodes, leafNodes, size-1, d_objects);
    checkCudaErrors(cudaGetLastError());
    
    cudaMemcpy(h_internalNodes, d_internalNodes, internal_size, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());
    
    iterativeTraversal(h_internalNodes);
/*    
    cudaMemcpy(d_internalNodes, h_internalNodes, internal_size, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    
    render<<<blocks, nthreads>>>(d_frameBuffer, nx, ny, ns, d_cam, &d_internalNodes[0], d_rand_state, depth);
    checkCudaErrors(cudaGetLastError());
*/
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
    
    exit(0);
    
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
  
	//free_world<<<1,1>>>(d_objects, size, d_world,d_cam);
	cudaFree(d_cam);
	//cudaFree(d_world);
	cudaFree(d_objects);
	cudaFree(d_rand_state);
	cudaFree(d_frameBuffer);
	
	cudaEventDestroy(E0); cudaEventDestroy(E1);
	
	cudaDeviceReset();  
}
