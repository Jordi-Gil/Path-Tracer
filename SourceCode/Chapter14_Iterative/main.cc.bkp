#include <iostream>
#include <cfloat>
#include <fstream>
#include <string>
#include <sys/time.h>

#include <bitset>

#include "Sphere.hh"
#include "MovingSphere.hh"
#include "Camera.hh"
#include "Material.hh"
#include "Node.hh"

double getusec_() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}

#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME stamp = getusec_() - stamp;\
                        stamp = stamp/1e6;

void error(const char *message){
  std::cout << message << std::endl;
  exit(0);
}

void help(){
  std::cout << "\n"  << std::endl;
  std::cout << "\t[-d] [--defult] Set the parameters to default values"  << std::endl;
  std::cout << "\t                (2048x1080) | AAit: 10 | depth: 10 | spheres: 4 " << std::endl;
  std::cout << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cout << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cout << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cout << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cout << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cout << "\t[-f][--file]    File name of pic generated." << std::endl;
  std::cout << "\t[-h][--help]    Show help." << std::endl;
  std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Examples of usage:" << std::endl;
  std::cout << "./path_tracing_sec -d"  << std::endl;
  std::cout << "./path_tracing_sec -sizeX 2000"<< std::endl;
  exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &filename){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 2048; ny = 1080; ns = 10; depth = 10; dist = 4; filename = "pic.ppm";
  
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
  		dist = atoi(argv[i+1]);
  		if(depth == 0) error("-spheres value expected or cannot be 0");
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

Node *generateHierarchy(Hitable **list, int numberObj) {
    
    
    Node* leafNodes = new Node[numberObj];
    Node* internalNodes = new Node[numberObj - 1];
    
    internalNodes[0].obj = NULL;
    
    for(int i = 0; i < numberObj; i++) 
        leafNodes[i].obj = list[i];
    
    return &internalNodes[0];
}

Node *random_scene(int dist){
    dist = 0;
  
    int n = (2*dist)*(2*dist) + 5;
  
    Hitable **list = new Hitable*[n+1];
  
    list[0] = new Sphere(Vector3(0,-1000,0),1000, new Lambertian(Vector3(0.5,0.5,0.5)));
  
    int i = 1;
  
    for(int a = -dist; a < dist; a++){
        for(int b = -dist; b < dist; b++){
            float choose_mat = (rand()/(RAND_MAX + 1.0));
            Vector3 center(a+0.9*(rand()/(RAND_MAX + 1.0)), 0.2, b+0.9*(rand()/(RAND_MAX + 1.0)));
      
            if((center-Vector3(0,0,0)).length() > 0.995){
                if(choose_mat < 0.8){ //diffuse
                    list[i++] = new MovingSphere(center, center+Vector3(0,0.5*(rand()/(RAND_MAX + 1.0)),0), 0.0, 1.0, 0.2, 
                        new Lambertian(Vector3(
                            (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                            (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                            (rand()/(RAND_MAX + 1.0)))));
                }
                else if(choose_mat < 0.95){ //metal
                    list[i++] = new Sphere(center, 0.2, new Metal(Vector3(
                        0.5*(1+(rand()/(RAND_MAX + 1.0))),
                        0.5*(1+(rand()/(RAND_MAX + 1.0))),
                        0.5*(1+(rand()/(RAND_MAX + 1.0)))
                    ),  0.5*(rand()/(RAND_MAX + 1.0))));
                }
                else{
                    list[i++] = new Sphere(center, 0.2, new Dielectric(Vector3::One(),1.5));
                }
            }
        }
    }
  
    list[i++] = new Sphere(Vector3(0,1,0), 1.0, new Dielectric(Vector3::One(),1.5));
    list[i++] = new Sphere(Vector3(-4,1,0),1.0, new Lambertian(Vector3(0.4,0.2,0.1)));
    list[i++] = new Sphere(Vector3(4,1,0),1.0, new Metal(Vector3(0.7,0.6,0.5),0.0));
  
    list[i++] = new Sphere(Vector3( 4, 1, 5), 1.0, new Metal(Vector3(0.9, 0.2, 0.2),0.0));
  
    Node *root = generateHierarchy(list, i);
    
    return root;
    
    //return new BVH_node(list,i,0,1,"init",0);
}

Vector3 color(const Ray& ray, Node *world, int depth, int const _depth){
    hit_record rec;
    if(world->checkCollision(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        if(depth < _depth && rec.mat_ptr->scatter(ray, rec, attenuation, scattered)){
            return attenuation*color(scattered, world, depth+1, _depth);
        }
        else return Vector3::Zero();
    }
    else{
        Vector3 unit_direction = unit_vector(ray.direction());
        float t = 0.5 * (unit_direction.y() + 1.0);
        return (1.0-t) * Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
    }
}

static unsigned int sortedMortonCodes[11] = {1,2,3,8,10,16,24,36,192,193,224};

int2 determineRange(int size, int idx) {
    
    int numberObj = size-1;
    
    if(idx == 0) return int2(0,numberObj);
    
    unsigned int idxCode = sortedMortonCodes[idx];
    unsigned int idxCodeUp = sortedMortonCodes[idx+1];
    unsigned int idxCodeDown = sortedMortonCodes[idx-1];
    
    int prefixUp = Helper::clz32d(idxCode ^ idxCodeUp);
    int prefixDown = Helper::clz32d(idxCode ^ idxCodeDown);
    
    //std::cout << "El bit que difiere " << idx << " con " << idx+1 << " es " << prefixUp << std::endl;
    //std::cout << "El bit que difiere " << idx << " con " << idx-1 << " es " << prefixDown << std::endl;
    
    int d = Helper::sgn( prefixUp - prefixDown );
    
    //std::cout << "El signo es: " << d << std::endl;
    
    int dmin;
    
    if(d < 0) dmin = prefixUp;
    else if (d > 0) dmin = prefixDown;
    
    //std::cout << "dmin entonces es: " << dmin << std::endl;
    
    int lmax = 2;
    
    int newBoundary;
    int bitPrefix;
    do {
        //std::cout << "lmax es " << lmax << std::endl;
        newBoundary = idx + lmax * d;
        bitPrefix = -1;
        //std::cout << "Movemos el limite a " << newBoundary;
        if(newBoundary >= 0 and newBoundary <= numberObj){
            unsigned int newCode = sortedMortonCodes[idx + lmax * d];
            
            bitPrefix = Helper::clz32d(idxCode ^ newCode);
            //std::cout << " Con prefijo " << bitPrefix << std::endl;
            if(bitPrefix > dmin) lmax *= 2;
            
        } //else std::cout << std::endl;
        
        
    } while(bitPrefix > dmin);
    
    //std::cout << "lmax al final vale " << lmax << std::endl;
    
    int l = 0;
    
    for(int t = lmax/2; t >= 1; t /= 2){
        
        int newUpperBound = idx + (l + t) * d;
        
        if(newUpperBound <= numberObj and newUpperBound >= 0){
            unsigned int splitCode = sortedMortonCodes[newUpperBound];
            int splitPrefix = Helper::clz32d(idxCode ^ splitCode);
            
            if(splitPrefix > dmin) l += t;
        }
        
    }
    
    
    //std::cout << "l vale " << l << std::endl;
    
    int jdx = idx + l * d;
    
    if(jdx < idx) return int2(jdx,idx);
    else return int2(idx,jdx);
    
}

unsigned int split(int first, int last) {
    
    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];
    
    //std::bitset<32> b1(firstCode); std::bitset<32> b2(lastCode);
    //std::cout << b1 << "\n" << b2 << std::endl;
    
    if(firstCode == lastCode)
        return (first + last) >> 1;
     
    
    int commonPrefix = Helper::clz32d(firstCode ^ lastCode);
    
    //std::cout << "Bit de diferencia: " << commonPrefix + 1<< std::endl;
    
    int split = first;
    //std::cout << "Corte: " << split << std::endl;
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        //std::cout << "Extremo superior: "  << step << std::endl;
        
        int newSplit = split + step; 
        //std::cout << "Avanzamos con " << newSplit << " ? " <<  newSplit << " < " << last << std::endl;
        if(newSplit < last){
            //std::cout << "Avanzamos" << std::endl;
            unsigned int splitCode = sortedMortonCodes[newSplit];
            
            int splitPrefix = Helper::clz32d(firstCode ^ splitCode);
            
            //std::bitset<32> b3(splitCode);
            //std::cout << b1 << "\n" << b3 << std::endl;
            
            //std::cout << "Bit de diferencia: " << splitPrefix + 1 << std::endl;
            //std::cout << "Avanzamos el corte? " << splitPrefix << " > " << commonPrefix << std::endl;
            //Si el nuevo corte 
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
                //std::cout << "Avanzamos el corte a " << split << std::endl;
            }
            //else std::cout << "No" << std::endl;
        }
        //else std::cout << "No avanzamos" << std::endl;
        
    } while (step > 1);
    
    return split;
        
}

unsigned int prueba(int size) {
    
    std::cout << "Size --> " << size << std::endl;
    std::cout << "idx from 0 to < " << size - 1 << std::endl;
    for (int idx = 0; idx < size-1; idx++) {
        int2 range = determineRange(size,idx);
        std::cout << "Range [" << range.x << "," << range.y << "] ";
    
        int first = range.x;
        int last = range.y;
        
        int spliti = split(range.x, range.y);
        
        std::cout << "Split en  " << spliti << std::endl;
        bool ha = false; bool hb = false;
        if(spliti == first){
            ha = true;
            std::cout << "childA: hoja" << std::endl;
        }
        else std::cout << "childA: internal" << std::endl;
        
        if(spliti+1 == last) {
            hb = true;
            std::cout << "childB: hoja" << std::endl;
        }
        else std::cout << "childB: internal" << std::endl;
        
        std::cout << "Nodo interno " << idx << " tiene de hijoA a " << spliti << std::endl;
        std::cout << "Nodo interno " << idx << " tiene de hijoB a " << spliti+1 << std::endl;
        if(ha) std::cout << "El nodo hoja ";
        else std::cout << "El nodo interno ";
        std::cout << spliti << " tiene de padre a " << idx << std::endl;
        if(hb) std::cout << "El nodo hoja ";
        else std::cout << "El nodo interno ";
        std::cout << spliti+1 << " tiene de padre a " << idx << std::endl;
        std::cout << std::endl;
    }
    return -1;
}

int main(int argc, char **argv)
{
    
    if(atoi(argv[1]) == 0) prueba(atoi(argv[2]));
    else if(atoi(argv[1]) == 1){
        std::cout << "Range [" << atoi(argv[2]) << "," << atoi(argv[3]) << "] ";
        int spliti = split(atoi(argv[2]),atoi(argv[3]));
        std::cout << "Split en  " << spliti << std::endl;
        spliti = 0;
    }
    else if(atoi(argv[1]) == 2) {
        std::cout << "Index " << atoi(argv[3]) << std::endl;
        int2 range = determineRange(atoi(argv[2]),atoi(argv[3]));
        std::cout << "Range [" << range.x << "," << range.y << "] " << std::endl;
        int splitX = split(range.x, range.y);
        std::cout << "Split: " << splitX << std::endl;
        
        splitX = 0;
    }
    else if(atoi(argv[1]) == 3) {
        unsigned int down = sortedMortonCodes[atoi(argv[2])];
        unsigned int pos = sortedMortonCodes[atoi(argv[2])];
        unsigned int up = sortedMortonCodes[atoi(argv[2])+1];
        
        int posUp = Helper::clz32d(pos ^ up);
        int posDw = Helper::clz32d(pos ^ down);
        
        std::cout << "Bit de ki vs ki+1 " << posUp << std::endl;
        std::cout << "Bit de ki vs ki-1 " << posDw << std::endl;
        std::cout << "Signo " << Helper::sgn(posUp - posDw) << std::endl;
    }

    exit(0);
    
  int nx, ny, ns, depth, dist;
  std::string filename;
  
  parse_argv(argc, argv, nx, ny, ns, depth, dist, filename);
  
  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels" << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << n << " spheres." << std::endl;
  
  std::ofstream pic;
  pic.open(filename.c_str());
  
  pic << "P3\n" << nx << " " <<  ny << "\n255" << "\n";
  
  
  Node *world = random_scene(dist);
  
  exit(0);
  
  Vector3 lookfrom(13,2,3);
  Vector3 lookat(0,0,0);
  float dist_to_focus = 10.0;
  float aperture = 0.1;

  Camera cam(lookfrom, lookat, Vector3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
  
  std::cout << "Creating image..." << std::endl;
  double stamp;
  START_COUNT_TIME;
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
        
      Vector3 col = Vector3::Zero();
      
      for(int s = 0; s < ns; s++){
        float u = float(i + (rand()/(RAND_MAX + 1.0))) / float(nx);
        float v = float(j + (rand()/(RAND_MAX + 1.0))) / float(ny);
        
        Ray r = cam.get_ray(u, v);
        
        col += color(r, world, 0, depth);
      }
      
      col /= float(ns);
      col = Vector3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      pic << ir << " " << ig << " " << ib << "\n";
    }
  }
  pic.close();
  STOP_COUNT_TIME;
  std::cout << "Image created in " << stamp << " seconds" << std::endl;
}


/*
 * int seconds, hours, minutes;
cin >> seconds;
minutes = seconds / 60;
hours = minutes / 60;
cout << seconds << " seconds is equivalent to " << int(hours) << " hours " << int(minutes%60) 
     << " minutes " << int(seconds%60) << " seconds.";
*/
