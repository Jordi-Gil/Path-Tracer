#include "Scene.hh"

Camera loadCamera(const std::string &line, int nx, int ny) {
  
  char ch = ' ';
  size_t pos = line.find(ch);
  size_t init = 0;
  int countV3 = 0;
  int countIdx = 0;
  int count = 0;
  int countF = 0;
  std::string par;
  
  float *paramF = new float[3];
  Vector3 *paramV3 = new Vector3[3];
  
  while(pos != std::string::npos) {
    
    par = line.substr(init, pos-init);
    init = pos + 1;
    pos = line.find(ch, init);
    
    if(countV3 < 3 and count < 9) {
			if(countIdx < 3) {
				paramV3[countV3][countIdx] = stof(par);
				++countIdx;
			}
			else{
				countIdx = 0;
				++countV3;
				if(countV3 < 3) {
					paramV3[countV3][countIdx] = stof(par);
					++countIdx;
				}
			}
			++count;
		}
		else if(countF < 3){
      paramF[countF] = stof(par);
      ++countF;
    }
  }
  
  par = line.substr(init, std::min(pos,line.size())-init+1);
  paramF[countF] = stof(par);
  ++countF;
  
  return Camera(paramV3[0], paramV3[1], paramV3[2], paramF[0], float(nx)/float(ny), paramF[1], paramF[2], 0.0, 1.0);
}

Material loadMaterial(const std::string &line,int type) {
  
  std::stringstream ssin(line);  
  std::string par;
  
  int albedoCount = 0;
  float fuzz = -1.0;
  float ref_idx = -1.0;
  Vector3 albedo;
  
  while(ssin >> par) {
  
    if(albedoCount < 3){
      albedo[albedoCount] = stof(par);
      albedoCount++;
    } else {
      if(type == METAL) fuzz = stof(par);
      else if(type == DIELECTRIC) ref_idx = stof(par);
    }
  }
  return Material(type,albedo,fuzz,ref_idx);
}

Triangle loadTriangle(const std::string &line, int num) {

  std::stringstream ssin(line);
  std::string par;
  
  int vertexCount = 0;
  int trCount = 0;
  int count  = 0;

  Vector3 *position = new Vector3[3];
  Material mat;

  while(ssin >> par) {

    if(trCount < 3 and count < 9) {
      if(vertexCount < 3) {
        position[trCount][vertexCount] = stof(par);
        ++vertexCount;
      }
      else{
        vertexCount = 0;
        ++trCount;
        if(trCount < 3) {
          position[trCount][vertexCount] = stof(par);
          ++vertexCount;
        }
      }
      ++count;
    }
    else {
      if(par == "L") mat = loadMaterial(line.substr(line.find(par)+par.size()),LAMBERTIAN);
      else if(par == "M") mat = loadMaterial(line.substr(line.find(par)+par.size()),METAL);
      else if(par == "D") mat = loadMaterial(line.substr(line.find(par)+par.size()),DIELECTRIC);
      else if(par == "DL") mat = loadMaterial(line.substr(line.find(par)+par.size()),DIFFUSE_LIGHT);
    }
  }
  return Triangle("manual_"+std::to_string(num),position[0],position[1],position[2], mat);
}

void print(Triangle *tl, int n){
  std::cout << "print Size " << n << std::endl;
  for(int i = 0; i < n; i++){
    std::cout << "Triangle " << i << std::endl;
    for (int j = 0; j < 3; j++) {
      std::cout << tl[i][j] << std::endl;
    }
    std::cout << std::endl;
  }
}

mat4 getTransformMatrix(const std::vector<std::string> &transforms, const Vector3 &center) {
 
  mat4 transform = mat4::identity();
  
  for(int i = 0; i < transforms.size(); i ++) {
    
    std::stringstream ssin(transforms[i]);
    std::string par;
    
    while(ssin >> par) {
      if(par == "t") {
        ssin >> par;
        if(par == "center") {
          transform = math::translate(transform, -center);
        }
        else {
          Vector3 pos;
          pos[0] = stof(par);
          ssin >> par; pos[1] = stof(par);
          ssin >> par; pos[2] = stof(par);
          
          transform = math::translate(transform, pos);
        }
      }
      else if(par == "r") {
        ssin >> par; float ang = stof(par);
        Vector3 axis;
        ssin >> par; axis[0] = stof(par);
        ssin >> par; axis[1] = stof(par);
        ssin >> par; axis[2] = stof(par);
        
        transform = math::rotate(transform, ang, axis);
      }
      else if(par == "s") {
        ssin >> par; 
        if(par == "u") {
          ssin >> par;
          float s = stof(par);
          transform = math::scale(transform, s);
        }
        else if(par == "nu") {
          Vector3 scaled;
          ssin >> par; scaled[0] = stof(par);
          ssin >> par; scaled[1] = stof(par);
          ssin >> par; scaled[2] = stof(par);
          transform = math::scale(transform, scaled);
        }
      }
    }
  }
  return transform;
}

Obj loadObj(const std::string &line) {
  
  std::stringstream ssin(line);
  std::string par;
  
  int type = -1;
  bool loaded = false;
  Obj object;
  
  std::vector<std::string> transforms;
  
  Material mat;
  bool material = false;
  
  std::string name = "";
  
  while(ssin >> par) {
    
    std::string trans = "";
    
    if(par == "t") {
      trans += par+" ";
      ssin >> par;
      if(par == "center") {
        trans += par;
        transforms.push_back(trans);
      }
      else {
        trans += par+" ";
        ssin >> par; trans += par+" ";
        ssin >> par; trans += par;
        transforms.push_back(trans);
      }
    }
    else if(par == "r") {
      trans += par+" ";
      ssin >> par; trans += par+" ";
      ssin >> par; trans += par+" ";
      ssin >> par; trans += par+" ";
      ssin >> par; trans += par;
      transforms.push_back(trans);
    }
    else if(par == "s") {
      trans += par+" ";
      ssin >> par; 
      if(par == "u") {
        trans += par+" ";
        ssin >> par;
        trans += par;
      }
      else if(par == "nu") {
        ssin >> par; trans += par+" ";
        ssin >> par; trans += par+" ";
        ssin >> par; trans += par;
      }
      transforms.push_back(trans);
    }
    else if(par == "m") {
      ssin >> par;
      if(par == "L") { 
        mat = loadMaterial(line.substr(line.find(par)+par.size()+1),LAMBERTIAN);
        ssin >> par; ssin >> par; ssin >> par;
      }
      else if(par == "M") { 
        mat = loadMaterial(line.substr(line.find(par)+par.size()+1),METAL);
        ssin >> par; ssin >> par; ssin >> par; ssin >> par;
      }
      else if(par == "D") { 
        mat = loadMaterial(line.substr(line.find(par)+par.size()+1),DIELECTRIC);
        ssin >> par; ssin >> par; ssin >> par; ssin >> par;
      }
      else if(par == "DL") {
        mat = loadMaterial(line.substr(line.find(par)+par.size()+1),DIFFUSE_LIGHT);
        ssin >> par; ssin >> par; ssin >> par;
      }
      material = true;
    }
    else {
      if(par == "0"){
        type = OBJF;
      }
      else if(par == "1") {
        type = TRIF;
      }
      else {
        if(type < 0 or type > 1) throw std::runtime_error("Invalid type file");
        else if(!loaded){
          if(type == OBJF) name  = par + ".obj";
          else if(type == TRIF) name = par + ".tri";
          loaded = true;
        }
      }
    }
  }
  
  object = (material) ? Obj(type, name, true, mat) : Obj(type, name);
  
  object.applyGeometricTransform(getTransformMatrix(transforms, object.getObjCenter()));
  
  return object;
  
}

Sphere loadSphere(const std::string &line) {
  
  std::stringstream ssin(line);
  std::string par;
  
  int centerCount = 0;
  float radius = 0.0;
  
  Vector3 center;
  Material mat;
  
  while(ssin >> par) {
    
    if(centerCount < 4){ 
      if(centerCount < 3) center[centerCount] = stof(par);
      else radius = stof(par);
      centerCount++;
    }
    else {
      if(par == "L") mat = loadMaterial(line.substr(line.find(par)+par.size()),LAMBERTIAN);
      else if(par == "M") mat = loadMaterial(line.substr(line.find(par)+par.size()),METAL);
      else if(par == "D") mat = loadMaterial(line.substr(line.find(par)+par.size()),DIELECTRIC);
      else if(par == "DL") mat = loadMaterial(line.substr(line.find(par)+par.size()),DIFFUSE_LIGHT);
    }
  }
  return Sphere(center,radius,mat);
}

int *loadSizes(const std::string &line, int *s) {

  std::stringstream ssin(line);
  std::string par;
  
  s[0] = -1; s[1] = -1; s[2] = -1;
  int count = 0;
  
  while(ssin >> par) {
    
    if(par == "ns") count = 0;
    else if(par == "nt") count = 1;
    else if(par == "no") count = 2;
    else if(par[0] >= '0' and par[0] <= '9') s[count] = stoi(par);
    
  }
  
  if(s[0] == -1) s[0] = 0; 
  if(s[1] == -1) s[1] = 0; 
  if(s[2] == -1) s[2] = 0;
  return s;
}

void Scene::loadScene(int loadType, const std::string &filename){
  
  std::cout << "Loading scene...\n" << std::endl;
  
  if(loadType == FFILE) sceneFromFile(filename);
  else if(loadType == RANDOM) sceneRandom();
  else if(loadType == TRIANGL) sceneTriangle();
  
  std::cout << "Scene loaded\n" << std::endl;
}

void Scene::sceneFromFile(const std::string &filename) {
  std::cout << "Scene file from " << filename << std::endl;
  
  std::ifstream file("Scenes/"+filename);
  std::string line;
  
  int num_sp = 0;
  int num_tr = 0;
  int num_ob = 0;
  int num_ob_tr = 0;
  
  int *sizes = nullptr;
  
  if(file.fail()) throw std::runtime_error("Something goes wrong");
  
  Sphere *list_0;
  Triangle *list_1;
  Obj *list_3;
  
  Vector3 *max_list;
  Vector3 *min_list;
  
  Vector3 max(MIN);
  Vector3 min(INF);
  
  while(std::getline(file, line)) {
    
    std::string aux;
    std::stringstream ssin(line);
    ssin >> aux;
    
    if(aux == "ns") {
      sizes = new int[3];
      loadSizes(line, sizes);
      list_0 = new Sphere[sizes[0]];
      list_1 = new Triangle[sizes[1]];
      list_3 = new Obj[sizes[2]];
      max_list = new Vector3[sizes[2]];
      min_list = new Vector3[sizes[2]];
    }
    else if(aux == "0") {
      cam = loadCamera(line.substr(2,line.size()), nx, ny);
    }
    else if(aux == "1") {
      if(sizes != nullptr and num_sp < sizes[0]){
        list_0[num_sp] = loadSphere(line.substr(2,line.size()));
        compare(max, min, list_0[num_sp].getCenter());
        ++num_sp;
      }
      else std::cout << "MAX size exceeded" << std::endl;
    }
    else if(aux == "2") {
      if(sizes != nullptr and num_tr < sizes[1]) {
        list_1[num_tr] = loadTriangle(line.substr(2,line.size()),num_tr);
        compare(max, min, list_1[num_tr].getCentroid());
        ++num_tr;
      }
      else std::cout << "MAX size exceeded" << std::endl;
    }
    else if(aux == "3") {
      if(sizes != nullptr and num_ob < sizes[2]){
        list_3[num_ob] = loadObj(line.substr(2,line.size()));
        max_list[num_ob] = list_3[num_ob].getMax();
        min_list[num_ob] = list_3[num_ob].getMin();
        num_ob_tr += list_3[num_ob].getSize();
        num_ob++;
      }
    }
  }
  
  int num_objects = num_tr + num_ob_tr;
  
  std::cout << "Real size " << num_objects << std::endl;
  
  Triangle *list_aux = new Triangle[num_objects];//(Triangle *) malloc((num_objects) * sizeof(Triangle));
  
  for(int i = 0; i < num_tr; i++) std::cout << list_1[i].idx << std::endl;
  
  std::copy(list_1, list_1 + num_tr, list_aux);
  
  //free(list_1);
  
  int p = num_tr;
  
  for(int idx = 0; idx < num_ob; idx++) {
    compare(max,min,max_list[idx]);
    compare(max,min,min_list[idx]);
    
    int n = list_3[idx].getSize();
    std::copy(list_3[idx].getTriangles(), list_3[idx].getTriangles() + n, list_aux + p);
    p += n;
    
  }
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
  for(int idx = 0; idx < num_objects; idx++) {
      
    Vector3 point = list_aux[idx].getCentroid();
    
    point[0] = ((point[0] - min_x)/(max_x - min_x));
    point[1] = ((point[1] - min_y)/(max_y - min_y));
    point[2] = ((point[2] - min_z)/(max_z - min_z));
    
    list_aux[idx].setMorton(Helper::morton3D(point[0],point[1],point[2]));
  }

  std::cout << "Max " <<  max << " Min " << min << std::endl;
  
  if(num_objects > 0) {
    objects = new Triangle[num_objects];
    std::copy(list_aux, list_aux + num_objects, objects);
    std::sort(objects, objects + num_objects, ObjEval());
    size = num_objects;
  }
  
  std::cout << "scene Loaded" << std::endl;

}

void Scene::sceneRandom() {
  
  std::cout << "Random scene using dist = " << dist << std::endl;
  
  Vector3 lookfrom(20,40,-40);
  Vector3 lookat(0,0,0);
  float dist_to_focus = 40.0;
  float aperture = 0.1;

  cam = Camera(lookfrom, lookat, Vector3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0);
  
  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Max size " << n << std::endl;
  
  Sphere *list = new Sphere[n];
  
  Vector3 max(MIN);
  Vector3 min(INF);
  
  int objs = 0;
  
  list[objs] = Sphere(Vector3(0,-1000,0),1000, Material(LAMBERTIAN, Vector3(0.5,0.5,0.5)));
  compare(max, min, list[objs].getCenter());
  objs++;
  
  for(int a = -dist; a < dist; a++){
    for(int b = -dist; b < dist; b++){
      float choose_mat = (rand()/(RAND_MAX + 1.0));
      Vector3 center(a+0.9*(rand()/(RAND_MAX + 1.0)), 0.2, b+0.9*(rand()/(RAND_MAX + 1.0)));
      if((center-Vector3(0,0,0)).length() > 0.995){
        if(choose_mat < 0.8){ //diffuse
          
          list[objs] = Sphere(center, 0.2, Material(LAMBERTIAN, Vector3(
                                                            (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                                                            (rand()/(RAND_MAX + 1.0))*(rand()/(RAND_MAX + 1.0)), 
                                                            (rand()/(RAND_MAX + 1.0)))));
        }
        else if(choose_mat < 0.90){ //metal
            
          list[objs] = Sphere(center, 0.2, Material(METAL, Vector3(
                                                                0.5*(1+(rand()/(RAND_MAX + 1.0))),
                                                                0.5*(1+(rand()/(RAND_MAX + 1.0))),
                                                                0.5*(1+(rand()/(RAND_MAX + 1.0)))),
                                                            0.5*(rand()/(RAND_MAX + 1.0))
                                                          ));
        }
        else if(choose_mat < 0.95){
          list[objs] = Sphere(center, 0.2, Material(DIELECTRIC,Vector3::One(),-1.0,1.5));
        }
        else {
          list[objs] = Sphere(center, 0.2, Material(DIFFUSE_LIGHT, Vector3::One()));
        }
        compare(max, min, list[objs].getCenter());
        objs++;
      }
    }
  }
  
  list[objs] = Sphere(Vector3(0,1,0), 1.0, Material(DIELECTRIC,Vector3::One(),-1,1.5));
  compare(max, min, list[objs].getCenter()); objs++;
  
  list[objs] = Sphere(Vector3(-4,1,0),1.0, Material(LAMBERTIAN,Vector3(0.4,0.2,0.1)));
  compare(max, min, list[objs].getCenter()); objs++;
  
  list[objs] = Sphere(Vector3(4,1,0),1.0, Material(METAL,Vector3(0.7,0.6,0.5),0.0));
  compare(max, min, list[objs].getCenter()); objs++;
  
  list[objs] = Sphere(Vector3(4,1,5), 1.0, Material(METAL,Vector3(0.9, 0.2, 0.2),0.0));
  compare(max, min, list[objs].getCenter()); objs++;
  
  std::cout << "Real size " << objs << std::endl;
  std::cout << "Max " <<  max << " Min " << min << std::endl;
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
  for(int idx = 0; idx < objs; idx++) {
      
    Vector3 point = list[idx].getCenter();
    
    point[0] = ((point[0] - min_x)/(max_x - min_x));
    point[1] = ((point[1] - min_y)/(max_y - min_y));
    point[2] = ((point[2] - min_z)/(max_z - min_z));
    
    list[idx].setMorton(Helper::morton3D(point[0],point[1],point[2]));
  }
  
  spheres = new Sphere[objs];
  std::copy(list, list + objs, spheres);
  std::sort(spheres, spheres + objs, ObjEval2());
  
  
  size = objs;
  
}

void Scene::sceneTriangle() {
  
  std::cout << "Triangle scene" << std::endl;
  
  objects = new Triangle[2];

	Vector3 max(MIN);
  Vector3 min(INF);
	
	int objs = 0;
  
  objects[objs] = Triangle("random_1",Vector3(-4.0,0.0,0.0), Vector3(-2.0,0.0,0.5), Vector3(-3.0,-1.0,0.0) ,Material(LAMBERTIAN, Vector3(1.0,0.0,0.0)));
	compare(max, min, objects[0].getCentroid()); objs++;
	
	objects[objs] = Triangle("random_2",Vector3(0.0,0.0,0.0), Vector3(2.0,0.0,0.0), Vector3(1.0,-1.0,0.0) ,Material(METAL, Vector3(1.0,0.0,0.0)));
  compare(max, min, objects[1].getCentroid()); objs++;
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
  for(int idx = 0; idx < objs; idx++) {
    
    Vector3 point = objects[idx].getCentroid();
    
    point[0] = ((point[0] - min_x)/(max_x - min_x));
    point[1] = ((point[1] - min_y)/(max_y - min_y));
    point[2] = ((point[2] - min_z)/(max_z - min_z));
  
    objects[idx].setMorton(Helper::morton3D(point[0],point[1],point[2])+1);
    
  }
  
  std::sort(objects, objects + objs, ObjEval());
  size = 2;
}

Triangle *Scene::getObjects() {
  return objects;
}

unsigned int Scene::getSize() {
  return size;
}

Camera Scene::getCamera() {
  return cam;
}
