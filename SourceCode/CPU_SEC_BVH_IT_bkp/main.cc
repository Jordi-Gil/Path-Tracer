#include <iostream>
#include <cfloat>
#include <sys/time.h>
#include <utility>
#include <stack>
#include <queue>

#include "Camera.hh"
#include "Scene.hh"
#include "Node.hh"
#include "filters.hh"

#define STB_IMAGE_STATIC
#include "stb_image.h"

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

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &image, std::string &filename, bool &light, bool &random, bool &filter, int &diameterBi, float &gs, float &gr, int &diameterMean, int &diameterMedian, bool &skybox, bool &oneTex){
  
  if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
  nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; image = "image";
  filter = false; gs = 0; gr = 0; diameterBi = 11; diameterMean = 3; diameterMedian = 3;
  
  skybox = false; oneTex = false;
  light = true; random = true;
  
  bool imageName = false;

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

int LongestCommonPrefix(int i, int j, int numObjects, Triangle *d_list) {
  
  if(i < 0 or i > numObjects - 1 or j < 0 or j > numObjects - 1) return -1;
  
  int codeI = d_list[i].getMorton();
  int codeJ = d_list[j].getMorton();
  
  if(i == j) {
    printf("Equals Longest\n");
    return Helper::clz32d(codeI ^ codeJ);
  }
  else return Helper::clz32d(codeI ^ codeJ);
  
}

unsigned int findSplit(Triangle *d_list, int first, int last) {

  
  if(first == last){
      return -1;
    }
    
  int firstCode = d_list[first].getMorton();
  int lastCode = d_list[last].getMorton();
  
  int commonPrefix = Helper::clz32d(firstCode ^ lastCode);
  
  int split = first;
  int step = last - first;
  
  do {
      step = (step + 1 ) >> 1;
      int newSplit = split + step; 
      
      if(newSplit < last){
    
          int splitCode = d_list[newSplit].getMorton();
          
          int splitPrefix = Helper::clz32d(firstCode ^ splitCode);
    
          if(splitPrefix > commonPrefix){
              
              split = newSplit;
          }
      }
      
  } while (step > 1);
  
  return split;
        
}

int2 determineRange(Triangle *d_list, int idx, int objs) {
    
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
        
    if(jdx < idx) return int2(jdx,idx);
    else return int2(idx,jdx);
}

Node* generateHierarchy(Triangle *d_list, int numberObj) {
    
    Node* leafNodes = new Node[numberObj];
    Node* internalNodes = new Node[numberObj - 1];
    
    for(int idx = 0; idx < numberObj; idx++) {
        
        d_list[idx].setMorton(d_list[idx].getMorton()+idx);
        
        leafNodes[idx].obj = &d_list[idx];
    }

    for(int idx = 0; idx < numberObj - 1; idx++) {
        
        //determine range
        int2 range = determineRange(d_list, idx, numberObj-1);
        
        int first = range.x;
        int last = range.y;
        
        //find partition point
        int split = findSplit(d_list, first, last);
        
        if(split == -1){
          split = (first+last) >> 1;
          ++last;
        }
        
        Node *childA;
        Node *childB;
        
        if(split == first) childA = &leafNodes[split];
        else childA = &internalNodes[split];
        
        if (split + 1 == last) childB = &leafNodes[split + 1];
        else childB = &internalNodes[split + 1];
        
        internalNodes[idx].left = childA;
        internalNodes[idx].right = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
    }
    
    return &internalNodes[0];
}

Vector3 color(const Ray& ray, Node *world, int depth, bool light, bool skybox, int const _depth, Skybox sky, bool oneTex, unsigned char **textures) {
  hit_record rec;
  if(world->intersect(ray, 0.00001, FLT_MAX, rec)){
      Ray scattered;
      Vector3 attenuation;
      Vector3 emitted = rec.mat_ptr.emitted(rec.u, rec.v, oneTex, textures);
      if(depth < _depth and rec.mat_ptr.scatter(ray, rec, attenuation, scattered)){
          return emitted + attenuation*color(scattered, world, depth+1, light, skybox, _depth, sky, oneTex, textures);
      }
      else return emitted;
  }
  else{
    if(skybox && sky.hit(ray, 0.00001, FLT_MAX, rec)){
      return rec.mat_ptr.emitted(rec.u, rec.v, oneTex, textures);
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
}

Node *random_scene(int dist, int nx, int ny, bool random, const std::string filename, Camera &cam, Skybox &sky, bool oneTex, unsigned char **textures, int size) {
  
  Scene scene = Scene(dist, nx, ny);
  
  if(random) scene.loadScene(TRIANGL);
  else scene.loadScene(FFILE,filename, oneTex);
  
  size = scene.getSize();
  cam = scene.getCamera();
  sky = scene.getSkybox();
  if(oneTex){
    textures = scene.getTextures();
  }
  
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
    
  int nx, ny, ns, depth, dist, diameterBi, diameterMean, diameterMedian;
  bool light, random, filter, skybox, oneTex;
  float gs, gr;
  std::string filename, image;

  parse_argv(argc, argv, nx, ny, ns, depth, dist, image, filename, light, random, filter, diameterBi, gs, gr, diameterMean, diameterMedian, skybox, oneTex);
  
  Camera cam;
  Skybox sky;
  unsigned char **textures;
  int size;
  
  Scene scene = Scene(dist, nx, ny);
  
  if(random) scene.loadScene(TRIANGL);
  else scene.loadScene(FFILE,filename, oneTex);
  
  size = scene.getSize();
  cam = scene.getCamera();
  sky = scene.getSkybox();
  if(oneTex){
    textures = scene.getTextures();
  }
  
  double stamp;
  START_COUNT_TIME;
  
  Node *world = generateHierarchy(scene.getObjects(), scene.getSize());
  
  STOP_COUNT_TIME;
  std::cout << "BVH created in " << stamp << " seconds" << std::endl;
  
  computeBoundingBox(world);
  std::cout << "Bounding box computed" << std::endl;
  
  std::cout << "Creating " << image << " with (" << nx << "," << ny << ") pixels." << std::endl;
  std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
  std::cout << "The world have " << size << " objects." << std::endl;
  if(light) std::cout << "Ambient light ON" << std::endl;
  else std::cout << "Ambient light OFF" << std::endl;
  
  
  std::cout << "\n\nWorld created\n\n";

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

        col += color(r, world, 0, light, skybox, depth, sky, oneTex, textures);
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
  image = "../Resources/Images/CPU_BVH/"+image;
  
  stbi_write_png(image.c_str(), nx, ny, 3, data, nx*3);

  if(filter){
    std::cout << "Filtering image using bilateral filter with Gs = " << gs << " and Gr = " << gr << " and window of diameter " << diameterBi << std::endl;
    std::string filenameFiltered = image.substr(0, image.length()-4) + "_bilateral_filter.png";
    int sx, sy, sc;
    unsigned char *imageData = stbi_load(image.c_str(), &sx, &sy, &sc, 0);
    unsigned char *imageFiltered = new unsigned char[sx*sy*3];;
    
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
