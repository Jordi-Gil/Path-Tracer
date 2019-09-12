#include <iostream>
#include <fstream>
#include <string>
#include <cfloat>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>

#include "Sphere.cuh"
#include "MovingSphere.cuh"
#include "HitableList.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "BVH_node.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define Random (curand_uniform(&local_random))

void error(const char *message) {
  
	std::cout << message << std::endl;
	exit(0);
}

void help() {

	std::cout << "\n"  << std::endl;
	std::cout << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
	std::cout << "\t                size: (2048x1080) | AAit: 10 | depth: 10 | spheres: 4 | nthreads: 32"  << std::endl;
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
  
	nx = 2048; ny = 1080; ns = 10; depth = 10; dist = 4; nthreads = 32; filename = "pic.ppm"; numGPUs = 1;
  
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

__global__ void free_world(Hitable **d_list, Hitable **d_world, Camera **d_cam) {
  
	int n = (*d_world)->length();
	for(int i = 0; i < n; i++) {
		delete ((Sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	
	delete *d_world;
	delete *d_cam;
}

__global__ void create_world(Hitable **d_list, Hitable **d_world, Camera **d_cam, int nx, int ny, int dist, curandState *random){
  
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_random = *random;

		d_list[0] = new Sphere(Vector3(0,-1000,-1), 1000, new Lambertian(Vector3(0.5, 0.5, 0.5)));
    
		int i = 1;
		for (int a = -dist; a < dist; a++) {
			for (int b = -dist; b < dist; b++) {
				float material = Random;
				Vector3 center(a+0.9*Random, 0.2, b+0.9*Random);
	
				if ((center-Vector3(0,0,0)).length() > 0.995) {
					if (material < 0.8) d_list[i++] = new MovingSphere(center, center+Vector3(0,0.5*Random,0),0.0,1.0,.2,new Lambertian(Vector3(Random*Random, Random*Random, Random*Random)));
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
		
		//*d_world = new HitableList(d_list, i);
		*d_world = new BVH_node(d_list, i, 0, 1, &local_random, 0, "Init");
		
		Vector3 lookfrom(13,2,3);
		Vector3 lookat(0,0,0);
		Vector3 up(0,1,0);
		float dist_to_focus = 10; (lookfrom-lookat).length();
		float aperture = 0.1;
		*d_cam = new Camera(lookfrom, lookat, up, 20, float(nx)/float(ny), aperture, dist_to_focus,0.0,0.1);
	}
}

__device__ Vector3 color(const Ray& ray, Hitable **world, int depth, curandState *random){
  
	Ray cur_ray = ray;
	Vector3 cur_attenuation = Vector3(1.0,1.0,1.0);
	for(int i = 0; i < depth; i++){ 
		hit_record rec;
		if( (*world)->hit(cur_ray, 0.001, FLT_MAX, rec)) {
      
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

__global__ void rand_init(curandState *random, int seed) {
  
	if(threadIdx.x == 0 && blockIdx.x == 0){
		curand_init(seed, 0, 0, random);
	}
	
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state, unsigned long long seed, int minY, int maxY){
  
	int num = blockIdx.x*blockDim.x + threadIdx.x;
 
	int i = num%max_x;
	int j = num/max_x + minY;
  
	if( (i >= max_x) || (j >= max_y) ) return;
    
	int pixel_index = num + minY;
    
	curand_init((seed << 20) + pixel_index, 0, 0, &rand_state[pixel_index]);
  
}

__global__ void render(Vector3 *fb, int max_x, int max_y, int ns, Camera **cam, Hitable **world, curandState *d_rand_state, int depth, int minY, int maxY) {
	
	int num = blockIdx.x*blockDim.x + threadIdx.x;
 
	int i = num%max_x;
	int j = num/max_x + minY;
	
	curandState local_random;
      
	int pixel_index = num + minY;
	
	local_random = d_rand_state[pixel_index];
	
	Vector3 col(0,0,0);
	
	for(int s = 0; s < ns; s++){
	
		float u = float(i + Random) / float(max_x);
		float v = float(j + Random) / float(max_y);
		
		Ray r = (*cam)->get_ray(u, v, &local_random);
		col += color(r, world, depth, &local_random);
		
	}
	
	d_rand_state[pixel_index] = local_random;
	
	col /= float(ns);
	
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	
	fb[pixel_index - minY] = col;

}

int main(int argc, char **argv) {

	std::cout << "GPU Info " << std::endl;
  
	cudaSetDevice(0);
	int device;
	cudaGetDevice(&device);
    
	cudaDeviceProp properties;
	checkCudaErrors( cudaDeviceSetLimit( cudaLimitStackSize, 4000 ) );
	checkCudaErrors( cudaGetDeviceProperties( &properties, device ) );
	
	size_t limit;
	checkCudaErrors( cudaDeviceGetLimit( &limit, cudaLimitStackSize ) );
    
	if( properties.major > 3 || ( properties.major == 3 && properties.minor >= 5 ) )
	{
      
		std::cout << "Running on GPU " << device << " (" << properties.name << ")" << std::endl;
		std::cout << "Compute mode: " << properties.computeMode << std::endl;
		std::cout << "Concurrent Kernels: " << properties.concurrentKernels << std::endl;
		std::cout << "Warp size: " << properties.warpSize << std::endl;
		std::cout << "Major: " << properties.major << " Minor: " << properties.minor << std::endl;
		std::cout << "Cuda limit stack size: " << limit << "\n\n" << std::endl;
	}
	else std::cout << "GPU " << device << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;

    int count;
  
	checkCudaErrors(cudaGetDeviceCount(&count));
  
	cudaEvent_t E0, E1;
	cudaEventCreate(&E0); cudaEventCreate(&E1);

	float totalTime;
	
	int nx, ny, ns, depth, dist, nthreads, numGPUs;
	std::string filename;
  
	parse_argv(argc, argv, nx, ny, ns, depth, dist, nthreads, filename, numGPUs, count);

	int n = (2*dist)*(2*dist)+5;
  
	std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels with " << nthreads << " threads, using " << numGPUs << " GPUs." << std::endl;
	std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
	std::cout << "The world have maximum " << n << " spheres." << std::endl;

	/* Seed for CUDA Random */
	unsigned long long int seed = 1000;
	
	/* #pixels of the image */
	int num_pixels = nx*ny;
	int elementsToJump = num_pixels/numGPUs;
	int bytesToJump = elementsToJump * sizeof(Vector3);
	
	/* Host variables */
	float fb_size = num_pixels*sizeof(Vector3);
	float drand_size = num_pixels*sizeof(curandState);
	Vector3 *h_frameBuffer;
  
	int blocks = (nx * ny)/(numGPUs * nthreads);
	
	/* Allocate Memory Host */
	cudaMallocHost((Vector3**)&h_frameBuffer, fb_size);
	
	/* Device variables */
	Vector3 **d_frames = (Vector3 **) malloc(numGPUs*sizeof(Vector3));
	Hitable ***d_lists = (Hitable ***) malloc(numGPUs*sizeof(Hitable));
	Hitable ***d_worlds = (Hitable ***) malloc(numGPUs*sizeof(Hitable));
	Camera  ***d_cameras = (Camera ***) malloc(numGPUs*sizeof(Camera)); 
	curandState **d_randstates = (curandState **) malloc (numGPUs*sizeof(curandState));
	curandState **d_randstates2 = (curandState **) malloc (numGPUs*sizeof(curandState));
    
    /* Allocate memory on each device */
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		
		curandState *d_rand_state2;
		Vector3 *d_frameBuffer;
		curandState *d_rand_state;
		Hitable **d_list;
		Hitable **d_world;
		Camera **d_cam;
		
		cudaMalloc(&d_frameBuffer, fb_size/numGPUs);
		cudaMalloc(&d_rand_state, drand_size);
		cudaMalloc(&d_rand_state2, sizeof(curandState));
		cudaMalloc(&d_list, n*sizeof(Hitable *));
		cudaMalloc(&d_world, sizeof(Hitable *));
		cudaMalloc(&d_cam, sizeof(Camera *));
		
		d_frames[i] = d_frameBuffer;
		d_lists[i] = d_list;
		d_worlds[i] = d_world;
		d_cameras[i] = d_cam;
		d_randstates[i] = d_rand_state;
		d_randstates2[i] = d_rand_state2;
		
	}
	
	cudaEventRecord(E0,0);
	cudaEventSynchronize(E0);
	
	for (int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		
		rand_init<<<1,1>>>(d_randstates2[i], seed);
		checkCudaErrors(cudaGetLastError());
		create_world<<<1,1>>>(d_lists[i], d_worlds[i], d_cameras[i], nx, ny, dist, d_randstates2[i]);
		checkCudaErrors(cudaGetLastError());
		render_init<<<blocks, nthreads>>>(nx, ny, d_randstates[i], seed, i*(ny/numGPUs), ((i+1)*(ny/numGPUs)));
		checkCudaErrors(cudaGetLastError());
		render<<<blocks, nthreads>>>(d_frames[i], nx, ny, ns, d_cameras[i], d_worlds[i], d_randstates[i], depth, i*(ny/numGPUs), ((i+1)*(ny/numGPUs)));
		checkCudaErrors(cudaGetLastError());
		
  }
	
	for (int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		cudaMemcpyAsync(&h_frameBuffer[elementsToJump*i], d_frames[i], bytesToJump, cudaMemcpyDeviceToHost);
		
	}
	
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		cudaDeviceSynchronize();
	}
	
	cudaSetDevice(0);
	cudaEventRecord(E1,0);
	cudaEventSynchronize(E1);
	cudaEventElapsedTime(&totalTime,E0,E1);
  
	std::cout << "Total Time: " << totalTime << " milisegs."<< std::endl; 
  
	std::cout << "Generating file image..." << std::endl;

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
  
	for (int i = 0; i < numGPUs; i++) {
		
		cudaSetDevice(i);
		
		free_world<<<1,1>>>(d_lists[i],d_worlds[i],d_cameras[i]);
		cudaFree(d_cameras[i]);
		cudaFree(d_worlds[i]);
		cudaFree(d_lists[i]);
		cudaFree(d_randstates[i]);
		
		cudaDeviceReset(); 
		
	}
  
	free(d_frames);
	free(d_lists);
	free(d_worlds);
	free(d_cameras);
	free(d_randstates);
	free(d_randstates2);
}