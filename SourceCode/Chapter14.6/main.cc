#include <iostream>
#include <cfloat>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <utility>
#include <stack>
#include <queue>
#include <limits>

#include "Camera.hh"
#include "Material.hh"
#include "Node.hh"

typedef std::vector< Sphere* > vh;

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
	std::cout << "\t[-light]        Turn on/off the ambient light. Values can be ON/OFF" << std::endl;
    std::cout << "\t[-f][--file]    File name of pic generated." << std::endl;
    std::cout << "\t[-h][--help]    Show help." << std::endl;
    std::cout << "\t                #spheres = (2*spheres)*(2*spheres) + 4" << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "Examples of usage:" << std::endl;
    std::cout << "./path_tracing_sec -d"  << std::endl;
    std::cout << "./path_tracing_sec -sizeX 2000"<< std::endl;
    exit(0);
  
}

void parse_argv(int argc, char **argv, int &nx, int &ny, int &ns, int &depth, int &dist, std::string &filename, bool &light){
  
    if(argc <= 1) error("Error usage. Use [-h] [--help] to see the usage.");
  
    nx = 1280; ny = 720; ns = 50; depth = 50; dist = 11; filename = "pic.ppm"; light = true;
  
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
        } else if(std::string(argv[i]) == "-light") {
					if((i+1) >= argc) error("-light value expected");
					if(std::string(argv[i+1]) == "ON") light = true;
					else if(std::string(argv[i+1]) == "OFF"){ 
						light = false; filename = "pic_off.ppm";
				}
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

unsigned int findSplit(const vh &sortedMortonCodes, int first, int last) {
    
    long long firstCode = sortedMortonCodes[first]->getMorton();
    long long lastCode = sortedMortonCodes[last]->getMorton();
    
    if(firstCode == lastCode)
        return (first + last) >> 1;
     
    int commonPrefix = Helper::clz32d(firstCode ^ lastCode);
    int split = first;
    
    int step = last - first;
    
    do {
        step = (step + 1 ) >> 1;
        int newSplit = split + step; 
        
        if(newSplit < last){
      
            long long splitCode = sortedMortonCodes[newSplit]->getMorton();
            
            int splitPrefix = Helper::clz32d(firstCode ^ splitCode);
      
            if(splitPrefix > commonPrefix){
                
                split = newSplit;
            }
        }
        
    } while (step > 1);
    
    return split;
        
}

int2 determineRange(const vh &sortedMortonCodes, int idx) {
    
    int numberObj = sortedMortonCodes.size()-1;
    
    if(idx == 0) return int2(0,numberObj);
    
    long long idxCode = sortedMortonCodes[idx]->getMorton();
    long long idxCodeUp = sortedMortonCodes[idx+1]->getMorton();
    long long idxCodeDown = sortedMortonCodes[idx-1]->getMorton();
    
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
                long long newCode = sortedMortonCodes[idx + lmax * d]->getMorton();
                
                bitPrefix = Helper::clz32d(idxCode ^ newCode);
                if(bitPrefix > dmin) lmax *= 2;
                
            }
            
        } while(bitPrefix > dmin);
        
        int l = 0;
        
        for(int t = lmax/2; t >= 1; t /= 2){
            
            int newUpperBound = idx + (l + t) * d;
            
            if(newUpperBound <= numberObj and newUpperBound >= 0){
                long long splitCode = sortedMortonCodes[newUpperBound]->getMorton();
                int splitPrefix = Helper::clz32d(idxCode ^ splitCode);
                
                if(splitPrefix > dmin) l += t;
            }
            
        }
        
        int jdx = idx + l * d;
        
        if(jdx < idx) return int2(jdx,idx);
        else return int2(idx,jdx);
    }
}

Node* generateHierarchy(const vh &sortedMortonCodes, int numberObj) {
    
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

Vector3 color(const Ray& ray, Node *world, int depth, bool light, int const _depth) {
    hit_record rec;
    if(world->checkCollision(ray, 0.001, MAXFLOAT, rec)){
        Ray scattered;
        Vector3 attenuation;
        Vector3 emitted = rec.mat_ptr.emitted();
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

Vector3 color_it(const Ray &ray, Node *world, int depth, bool light, int const _depth) {
	Ray cur_ray = ray;
	depth = 0*depth;
	Vector3 cur_attenuation = Vector3::One();
	for(int i = 0; i < _depth; i++) {
		hit_record rec;
		if(world->checkCollision(cur_ray, 0.001,FLT_MAX,rec)) {
			Ray scattered;
			Vector3 attenuation;
			Vector3 emitted = rec.mat_ptr.emitted();
			if(rec.mat_ptr.scatter(ray,rec,attenuation,scattered)) {
				cur_attenuation += emitted;
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else return emitted;
		}
		else {
			if(light){
				Vector3 unit_direction = unit_vector(cur_ray.direction());
				float t = 0.5*(unit_direction.y()+1.0);
				Vector3 c = (1.0-t)*Vector3::One()+t*Vector3(0.5,0.7,1.0);
				return cur_attenuation * c;
			}
			else return Vector3::Zero();
		}
	}
	return Vector3::Zero();
}

Node *random_scene(int dist) {
  
    vh list;
    
    Vector3 max(MIN);
    Vector3 min(INF);
    
    list.push_back(new Sphere(Vector3(0,-1000,0),1000, Material(LAMBERTIAN, Vector3(0.5,0.5,0.5))));
    
    compare(max, min, list.back()->getCenter());
    
    for(int a = -dist; a < dist; a++){
        for(int b = -dist; b < dist; b++){
            float choose_mat = (rand()/(RAND_MAX + 1.0));
            Vector3 center(a+0.9*(rand()/(RAND_MAX + 1.0)), 0.2, b+0.9*(rand()/(RAND_MAX + 1.0)));
            if((center-Vector3(0,0,0)).length() > 0.995){
                if(choose_mat < 0.8){ //diffuse
                    
                    list.push_back(new Sphere(center, 0.2, Material(LAMBERTIAN, Vector3(
                                                                      (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                                                                      (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                                                                      (rand()/(RAND_MAX + 1.0))))));
                }
                else if(choose_mat < 0.90){ //metal
                    
                    list.push_back(new Sphere(center, 0.2, Material(METAL, Vector3(
                                                                          0.5*(1+(rand()/(RAND_MAX + 1.0))),
                                                                          0.5*(1+(rand()/(RAND_MAX + 1.0))),
                                                                          0.5*(1+(rand()/(RAND_MAX + 1.0)))),
                                                                      0.5*(rand()/(RAND_MAX + 1.0))
                                                                    )));
                }
                else if(choose_mat < 0.95){
                    
                    list.push_back(new Sphere(center, 0.2, Material(DIELECTRIC,Vector3::One(),-1.0,1.5)));
                }
                else {
                  list.push_back(new Sphere(center, 0.2, Material(DIFFUSE_LIGHT, Vector3::One())));
                }
            }
            compare(max, min, list.back()->getCenter());
        }
    }
    
    list.push_back(new Sphere(Vector3(0,1,0), 1.0, Material(DIELECTRIC,Vector3::One(),-1,1.5)));
    compare(max, min, list.back()->getCenter());
    
    list.push_back(new Sphere(Vector3(-4,1,0),1.0, Material(LAMBERTIAN,Vector3(0.4,0.2,0.1))));
    compare(max, min, list.back()->getCenter());
    
    list.push_back(new Sphere(Vector3(4,1,0),1.0, Material(METAL,Vector3(0.7,0.6,0.5),0.0)));
    compare(max, min, list.back()->getCenter());
    
    list.push_back(new Sphere(Vector3(4,1,5), 1.0, Material(METAL,Vector3(0.9, 0.2, 0.2),0.0)));
    compare(max, min, list.back()->getCenter());
    
    std::cout << max << " " << min << std::endl;
    
    float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
    float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
    
    for(int i = 0; i < list.size(); i++) {
        
        Vector3 point = list[i]->getCenter();
        
        point[0] = ((point[0] - min_x)/(max_x - min_x));
        point[1] = ((point[1] - min_y)/(max_y - min_y));
        point[2] = ((point[2] - min_z)/(max_z - min_z));
        
        list[i]->setMorton(Helper::morton3D(point[0],point[1],point[2]));
    }
    vh list_aux = list;
    
    std::sort(list.begin(), list.end(), ObjEval());
    
    double stamp;
    START_COUNT_TIME;
    
    Node *root = generateHierarchy(list, list.size());
    
    STOP_COUNT_TIME;
    std::cout << "BVH created in " << stamp << " seconds" << std::endl;
    
    iterativeTraversal(root);
    
    return root;
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
	bool light;
	std::string filename;

	parse_argv(argc, argv, nx, ny, ns, depth, dist, filename,light);

	int n = (2*dist)*(2*dist)+5;
  
	std::cout << "Creating " << filename << " with (" << nx << "," << ny << ") pixels" << std::endl;
	std::cout << "With " << ns << " iterations for AntiAliasing and depth of " << depth << "." << std::endl;
	std::cout << "The world have " << n << " spheres." << std::endl;
	if(light) std::cout << "Ambient light ON" << std::endl;
	else std::cout << "Ambient light OFF" << std::endl;
    
    Node *world = random_scene(dist);
    
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
        
                col += color(r, world, 0, light, depth);
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
