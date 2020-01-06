#include <iostream>
#include <cfloat>
#include <sys/time.h>
#include <utility>
#include <stack>
#include <queue>

#include "Camera.hh"
#include "Scene.hh"
#include "Node.hh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
  std::cout << "\t                size: (1280x720) | AAit: 50 | depth: 50 | spheres: 11"  << std::endl;
  std::cout << "\t[-sizeX]        Size in pixels of coordinate X. Number greater than 0."  << std::endl;
  std::cout << "\t[-sizeY]        Size in pixels of coordinate Y. Number greater than 0."  << std::endl;
  std::cout << "\t[-AAit]         Number of iterations to calculate color in one pixel. Number greater than 0."  << std::endl;
  std::cout << "\t[-depth]        The attenuation of scattered ray. Number greater than 0."  << std::endl;
  std::cout << "\t[-spheres]      Factor number to calculate the number of spheres in the scene. Number greater than 0." << std::endl;
  std::cout << "\t[-light]        Turn on/off the ambient light. Values can be ON/OFF" << std::endl;
  std::cout << "\t[-i][--image]   File name of the pic generated." << std::endl;
  std::cout << "\t[-f][--file]    File name of the scene." << std::endl;
  std::cout << "\t[-h][--help]    Show help." << std::endl;
  std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Examples of usage:" << std::endl;
  std::cout << "./path_tracing_sec -d"  << std::endl;
  std::cout << "./path_tracing_sec -sizeX 2000 -f cornel" << std::endl;
  std::cout << "\n" << std::endl;
  format();
  exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &image, std::string &filename, bool &light, bool &random){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");

  nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; image = "random"; light = true; random = true;

  bool v_default = false;
  
  for(int i = 1; i < argc; i += 2) {

    if(v_default) error("Error usage. Use [-h] [--help] to see the usage.");

    if (std::string(argv[i]) == "-d" || std::string(argv[i]) == "--default") {
      if ((i+1) < argc) error("The default parameter cannot have more arguments.");
      std::cerr << "Default\n";
      v_default = true;
    } 
    else if (std::string(argv[i]) == "-sizeX"){
      if ((i+1) >= argc) error("-sizeX value expected");
      nx = atoi(argv[i+1]);
      if(nx == 0) error("-sizeX value expected or cannot be 0");
    } 
    else if (std::string(argv[i]) == "-sizeY") {
      if ((i+1) >= argc) error("-sizeY value expected");
      ny = atoi(argv[i+1]);
      if(ny == 0) error("-sizeY value expected or cannot be 0");
    } 
    else if (std::string(argv[i]) == "-AAit") {
      if ((i+1) >= argc) error("-AAit value expected");
      ns = atoi(argv[i+1]);
      if(ns == 0) error("-AAit value expected or cannot be 0");
    } 
    else if (std::string(argv[i]) == "-depth") {
      if ((i+1) >= argc) error("-depth value expected");
      depth = atoi(argv[i+1]);
      if(depth == 0) error("-depth value expected or cannot be 0");
    } 
    else if (std::string(argv[i]) == "-spheres") {
      if ((i+1) >= argc) error("-spheres value expected");
      dist = atoi(argv[i+1]);
      if(depth == 0) error("-spheres value expected or cannot be 0");
    } 
    else if (std::string(argv[i]) == "-i" || std::string(argv[i]) == "--image"){
      if ((i+1) >= argc) error("--image / -i value expected");
      image = std::string(argv[i+1]);
    }
    else if (std::string(argv[i]) == "-f" || std::string(argv[i]) == "--file"){
      if ((i+1) >= argc) error("--file / -f value expected");
      filename = std::string(argv[i+1]);
      image = filename;
      filename = filename+".txt";    
      random = false;
    } 
    else if(std::string(argv[i]) == "-light") {
      if((i+1) >= argc) error("-light value expected");
      if(std::string(argv[i+1]) == "ON") light = true;
      else if(std::string(argv[i+1]) == "OFF") light = false;
    } 
    else if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help" ){
      help();
    }
    else {
      error("Error usage. Use [-h] [--help] to see the usage.");
    }
  }
  if(!light) image = image+"_noktem";
  image = image+".png";
}

void computeBoundingBox(Node* root) {

    std::stack <Node *> stack;
    std::queue <Node *> queue;
    
    queue.push(root);
    
    while(!queue.empty()) {
        
        root = queue.front();

        queue.pop();
        stack.push(root);
        
        if(root->left) queue.push(root->left);
        if(root->right) queue.push(root->right);
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

unsigned int findSplit(Triangle *sortedMortonCodes, int first, int last) {
    
  long long firstCode = sortedMortonCodes[first].getMorton();
  long long lastCode = sortedMortonCodes[last].getMorton();
  
  if(firstCode == lastCode)
      return (first + last) >> 1;
    
  int commonPrefix = Helper::clz32d(firstCode ^ lastCode);
  int split = first;
  
  int step = last - first;
    
  do {
      step = (step + 1 ) >> 1;
      int newSplit = split + step; 
      
      if(newSplit < last){
    
          long long splitCode = sortedMortonCodes[newSplit].getMorton();
          
          int splitPrefix = Helper::clz32d(firstCode ^ splitCode);
    
          if(splitPrefix > commonPrefix){
              
              split = newSplit;
          }
      }
      
  } while (step > 1);
  
  return split;
        
}

int2 determineRange(Triangle *sortedMortonCodes, int idx, int numberObj) {
    
    if(idx == 0) return int2(0,numberObj);
    
    long long idxCode = sortedMortonCodes[idx].getMorton();
    long long idxCodeUp = sortedMortonCodes[idx+1].getMorton();
    long long idxCodeDown = sortedMortonCodes[idx-1].getMorton();
    
    if((idxCode == idxCodeDown) and (idxCode == idxCodeUp)) {
    
        int idxInit = idx;
        bool dif = false;
        while(!dif and idx > 0 and idx < numberObj){
            ++idx;
            if(idx >= numberObj) dif = true;
                
            if(sortedMortonCodes[idx].getMorton() != sortedMortonCodes[idx+1].getMorton()) dif = true;
        }
        
        return int2(idxInit, idx);
        
    } else {
        
        int prefixUp = Helper::clz32d(idxCode ^ idxCodeUp);
        int prefixDown = Helper::clz32d(idxCode ^ idxCodeDown);
        
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
                long long newCode = sortedMortonCodes[idx + lmax * d].getMorton();
                
                bitPrefix = Helper::clz32d(idxCode ^ newCode);
                if(bitPrefix > dmin) lmax *= 2;
                
            }
            
        } while(bitPrefix > dmin);
        
        int l = 0;
        
        for(int t = lmax/2; t >= 1; t /= 2){
            
            int newUpperBound = idx + (l + t) * d;
            
            if(newUpperBound <= numberObj and newUpperBound >= 0){
                long long splitCode = sortedMortonCodes[newUpperBound].getMorton();
                int splitPrefix = Helper::clz32d(idxCode ^ splitCode);
                
                if(splitPrefix > dmin) l += t;
            }
            
        }
        
        int jdx = idx + l * d;
        
        if(jdx < idx) return int2(jdx,idx);
        else return int2(idx,jdx);
    }
}

Node* generateHierarchy(Triangle *sortedMortonCodes, int numberObj) {
    
    Node* leafNodes = new Node[numberObj];
    Node* internalNodes = new Node[numberObj - 1];
    
    for(int idx = 0; idx < numberObj; idx++) {
        
        sortedMortonCodes[idx].setMorton(sortedMortonCodes[idx].getMorton()+idx);
        
        leafNodes[idx].obj = &sortedMortonCodes[idx];
    }

    for(int idx = 0; idx < numberObj - 1; idx++) {
        
        //determine range
        int2 range = determineRange(sortedMortonCodes, idx, numberObj-1);
        
        int first = range.x;
        int last = range.y;
        
        //find partition point
        int split = findSplit(sortedMortonCodes, first, last);
        
        //if(idx == 0) 
        //std::cout << "idx " << idx << " range [" << range.x << "," << range.y << "] split in " << split << " ";
        
        Node *childA;
        Node *childB;
        
        if(split == first){ childA = &leafNodes[split]; childA->id = "leaf_"+std::to_string(first);}
        else{
            childA = &internalNodes[split]; childA->id = "intr_"+std::to_string(split);
        }
        
        if (split + 1 == last) { childB = &leafNodes[split + 1]; childB->id = "leaf_"+std::to_string(last);}
        else{
            childB = &internalNodes[split + 1]; childB->id = "intr_"+std::to_string(split+1);
        }
        
        internalNodes[idx].id = "intr_"+std::to_string(idx);
        internalNodes[idx].left = childA;
        internalNodes[idx].right = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
        
        //if(idx == 0) 
        //std::cout << internalNodes[idx].id << "-->" << " LF: " << internalNodes[idx].left->id << " RG " << internalNodes[idx].right->id << std::endl;
        
    }
    
    return &internalNodes[0];
}

Vector3 color(const Ray& ray, Node *world, int depth, bool light, int const _depth) {
  hit_record rec;
  if(world->checkCollision(ray, 0.00001, MAXFLOAT, rec)){
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted(rec.u, rec.v);
      if(depth < _depth and rec.mat_ptr.scatter(ray, rec, attenuation, scattered)){
          return emitted + attenuation*color(scattered, world, depth+1,light, _depth);
      }
      else return emitted;
  }
  else{
  if(light) {
    Vector3 unit_direction = unit_vector(ray.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-t) * Vector3::One() + t*Vector3(0.5, 0.7, 1.0);
  }
  else return Vector3::Zero();
  }
}

Node *random_scene(int dist, int nx, int ny, bool random, const std::string filename, Camera &cam) {
  
  Scene scene = Scene(dist, nx, ny);
  
  if(random) scene.loadScene(TRIANGL);
  else scene.loadScene(FFILE,filename);
  
  cam = scene.getCamera();
  
  double stamp;
  START_COUNT_TIME;
  
  Node *root = generateHierarchy(scene.getObjects(), scene.getSize());
  
  STOP_COUNT_TIME;
  std::cout << "BVH created in " << stamp << " seconds" << std::endl;
  
  computeBoundingBox(root);
  
  std::cout << "Bounding box computed" << std::endl;
  
  return root;
}

int main(int argc, char **argv) {
    
  int nx, ny, ns, depth, dist;
  bool light, random;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, image, filename, light, random);

  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Creating " << image << " with (" << nx << "," << ny << ") pixels" << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  if(random) std::cout << "The world have " << n << " spheres max." << std::endl;
  else std::cout << "The world loaded via " << filename << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF\n" << std::endl;
  
  Camera cam;
  
  Node *world = random_scene(dist, nx, ny, random, filename, cam);
  
  std::cout << "\n\nWorld created\n\n";

  double stamp;
  START_COUNT_TIME;
  
  uint8_t *data = new uint8_t[nx*ny*3];
  
  std::cout << "Creating image..." << std::endl;
  
  int count = 0;
  
  for(int j = ny - 1; j >= 0; j--){
    for(int i = 0; i < nx; i++){
      
      Vector3 col = Vector3::Zero();

      for(int s = 0; s < ns; s++){
        float u = float(i + (rand()/(RAND_MAX + 1.0))) / float(nx);
        float v = float(j + (rand()/(RAND_MAX + 1.0))) / float(ny);

        Ray r = cam.get_ray(u, v);

        col += color(r, world, 0, light, depth);
      }
      
      col /= float(ns);
      col = Vector3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);
      
      data[count++] = ir;
      data[count++] = ig;
      data[count++] = ib;
      
    }
  }
  
  STOP_COUNT_TIME;
  std::cout << "Image created in " << stamp << " seconds" << std::endl;
  stbi_write_png(image.c_str(), nx, ny, 3, data, nx*3);
  
}


/*
 * int seconds, hours, minutes;
cin >> seconds;
minutes = seconds / 60;
hours = minutes / 60;
cout << seconds << " seconds is equivalent to " << int(hours) << " hours " << int(minutes%60) 
     << " minutes " << int(seconds%60) << " seconds.";
     
     p(Vector3(4,-4,-4));
  p(Vector3(4,4,-4));
  p(Vector3(4,4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(4,-4,-4));
  p(Vector3(4,4,4));
  p(Vector3(4,-4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,-4));
  p(Vector3(-4,4,-4));
  p(Vector3(4,4,-4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,-4));
  p(Vector3(4,4,-4));
  p(Vector3(4,-4,-4));
  std::cout << "\n" << std::endl;
  p(Vector3(4,4,-4));
  p(Vector3(-4,4,-4));
  p(Vector3(-4,4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(4,4,-4));
  p(Vector3(-4,4,4));
  p(Vector3(4,4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,4));
  p(Vector3(-4,4,4));
  p(Vector3(-4,4,-4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,4));
  p(Vector3(-4,4,-4));
  p(Vector3(-4,-4,-4));
  std::cout << "\n" << std::endl;
  p(Vector3(4,-4,4));
  p(Vector3(4,4,4));
  p(Vector3(-4,4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(4,-4,4));
  p(Vector3(-4,4,4));
  p(Vector3(-4,-4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,-4));
  p(Vector3(4,-4,-4));
  p(Vector3(4,-4,4));
  std::cout << "\n" << std::endl;
  p(Vector3(-4,-4,-4));
  p(Vector3(4,-4,4));
  p(Vector3(-4,-4,4));
  std::cout << "\n" << std::endl;
*/
