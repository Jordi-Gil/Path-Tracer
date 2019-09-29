#include <iostream>
#include <cfloat>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <utility>
#include <stack>
#include <queue>
#include <limits>

#include "Scene.hh"
#include "Camera.hh"
#include "Material.hh"
#include "Node.hh"

double getusec_() {
  struct timeval time;
  gettimeofday(&time, NULL);
  
  return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::min()
#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME stamp = getusec_() - stamp;\
                        stamp = stamp/1e6;               

Scene *scene = nullptr;
                        
void error(const char *message){
  std::cout << message << std::endl;
  exit(0);
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
  
    nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; filename = "pic.ppm";
  
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

void iterativeTraversal(Node* root) {

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

unsigned int findSplit(const Sphere *sortedMortonCodes, int first, int last) {
    
    unsigned int firstCode = sortedMortonCodes[first]->getMorton();
    unsigned int lastCode = sortedMortonCodes[last]->getMorton();
    
    if(firstCode == lastCode)
        return (first + last) >> 1;
     
    int commonPrefix = Helper::clz32d(firstCode ^ lastCode);
    int split = first;
    
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        int newSplit = split + step; 
        
        if(newSplit < last){
      
            unsigned int splitCode = sortedMortonCodes[newSplit]->getMorton();
            
            int splitPrefix = Helper::clz32d(firstCode ^ splitCode);
      
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
            }
        }
        
    } while (step > 1);
    
    return split;
        
}

int2 determineRange(const Sphere *sortedMortonCodes, int idx) {
    
    int numberObj = sortedMortonCodes.size()-1;
    
    if(idx == 0) return int2(0,numberObj);
    
    unsigned int idxCode = sortedMortonCodes[idx]->getMorton();
    unsigned int idxCodeUp = sortedMortonCodes[idx+1]->getMorton();
    unsigned int idxCodeDown = sortedMortonCodes[idx-1]->getMorton();
    if((idxCode == idxCodeDown) and (idxCode == idxCodeUp)) {
    
        int idxInit = idx;
        bool dif = false;
        while(!dif and idx > 0 and idx < numberObj){
            ++idx;
            if(idx >= numberObj) dif = true;
                
            if(sortedMortonCodes[idx]->getMorton() != sortedMortonCodes[idx+1]->getMorton()) dif = true;
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
                unsigned int newCode = sortedMortonCodes[idx + lmax * d]->getMorton();
                
                bitPrefix = Helper::clz32d(idxCode ^ newCode);
                if(bitPrefix > dmin) lmax *= 2;
                
            }
            
        } while(bitPrefix > dmin);
        
        int l = 0;
        
        for(int t = lmax/2; t >= 1; t /= 2){
            
            int newUpperBound = idx + (l + t) * d;
            
            if(newUpperBound <= numberObj and newUpperBound >= 0){
                unsigned int splitCode = sortedMortonCodes[newUpperBound]->getMorton();
                int splitPrefix = Helper::clz32d(idxCode ^ splitCode);
                
                if(splitPrefix > dmin) l += t;
            }
            
        }
        
        int jdx = idx + l * d;
        
        if(jdx < idx) return int2(jdx,idx);
        else return int2(idx,jdx);
    }
}

Node* generateHierarchy(const Sphere *sortedMortonCodes, int numberObj) {
    
    std::ofstream graph;
    graph.open("graph.dot");
    
    graph << "digraph BST { \n";
    graph << "\tnode [fontname=\"Arial\"];\n";
    
    Node* leafNodes = new Node[numberObj];
    Node* internalNodes = new Node[numberObj - 1];
    
    internalNodes[0].name = "i_0";
    
    for(int idx = 0; idx < numberObj; idx++) {
        
        sortedMortonCodes[idx]->setMorton(sortedMortonCodes[idx]->getMorton()+idx);
        
        leafNodes[idx].obj = sortedMortonCodes[idx];
        leafNodes[idx].name = "l_"+std::to_string(sortedMortonCodes[idx]->getMorton());
        graph << "\t" << leafNodes[idx].name << " [ label = \"" <<  leafNodes[idx].name.substr(2) << "\"];\n";
    }

    for(int idx = 0; idx < numberObj - 1; idx++) {
        
        //determine range
        
        graph << "\t" << "i_" << idx << " [ label = \"" << idx << "\"];\n";
        
        int2 range = determineRange(sortedMortonCodes, idx);
        
        int first = range.x;
        int last = range.y;
        
        //find partition point
        
        int split = findSplit(sortedMortonCodes, first, last);
        
        Node *childA;
        Node *childB;
        
        if(split == first) childA = &leafNodes[split];
        else{ 
            internalNodes[split].name = "i_"+std::to_string(split);
            childA = &internalNodes[split];
        }
        
        if (split + 1 == last) childB = &leafNodes[split + 1];
        else{
            internalNodes[split+1].name = "i_"+std::to_string(split+1);
            childB = &internalNodes[split + 1];
        }
        
        internalNodes[idx].name = "i_"+std::to_string(idx);
        
        internalNodes[idx].left = childA;
        internalNodes[idx].right = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
        
        graph << "\t" << internalNodes[idx].name << " -> {" << childA->name << " " << childB->name << "};\n";
        
    }
    graph << "}";
    graph.close();
    
    return &internalNodes[0];
}

void compare(Vector3 &max, Vector3 &min, Vector3 point) {
    
    if(point[0] > max[0]) max[0] = point[0]; //x
    if(point[1] > max[1]) max[1] = point[1]; //y
    if(point[2] > max[2]) max[2] = point[2]; //z
    
    if(point[0] < min[0]) min[0] = point[0]; //x
    if(point[1] < min[1]) min[1] = point[1]; //y
    if(point[2] < min[2]) min[2] = point[2]; //z
}

Vector3 color(const Ray& ray, Node *world, int depth, int const _depth) {
    hit_record rec;
    if(world->checkCollision(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        if(depth < _depth and rec.mat_ptr->scatter(ray, rec, attenuation, scattered)){
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

int depthTree(Node *root){
    
    if (root == NULL) return 0;
    else {
        
        int depth_L = depthTree(root->left);
        int depth_R = depthTree(root->right);
        
        if(depth_L > depth_R) return depth_L+1;
        else return depth_R+1;
        
    }
}

int main(int argc, char **argv) {
    
    int nx, ny, ns, depth, dist;
    std::string filename;
  
    parse_argv(argc, argv, nx, ny, ns, depth, dist, filename);
  
    int n = (2*dist)*(2*dist)+5;
  
    std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels" << std::endl;
    std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
    std::cout << "The world have " << n << " spheres." << std::endl;
  
    scene = new Scene();
    scene->loadScene();
    
    
  
    Vector3 lookfrom(13,2,3);
    Vector3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    Camera cam(lookfrom, lookat, Vector3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
  
    std::ofstream pic;
    pic.open(filename.c_str());
  
    pic << "P3\n" << nx << " " <<  ny << "\n255" << "\n";
    
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
