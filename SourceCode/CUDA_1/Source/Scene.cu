#include "Scene.cuh"

#define Random (rand()/(RAND_MAX + 1.0))

Material loadMaterial(const std::string &line, size_t &pos, char ch,int type) {

  pos = line.find(ch);
  size_t init = 0;
  
  std::string par;
  int albedoCount = 0;
  float fuzz = -1.0;
  float ref_idx = -1.0;
  Vector3 albedo;
  
  while(pos != std::string::npos) {
    par = line.substr(init, pos-init);
    init = pos + 1;
    pos = line.find(ch,init);
    
    if(albedoCount < 3){
      albedo[albedoCount] = stof(par);
      albedoCount++;
    }
  }
  
  par = line.substr(init, std::min(pos,line.size())-init+1);
  
  if(albedoCount < 3) albedo[albedoCount] = stof(par);
  else {
    if(type == METAL) fuzz = stof(par);
    else if(type == DIELECTRIC) ref_idx = stof(par);
  }
  
  return Material(type,albedo,fuzz,ref_idx);
}

Sphere loadSphere(const std::string &line) {
  
  char ch = ' ';
  size_t pos = line.find(ch);
  size_t init = 0;
  int centerCount = 0;
  float radius = 0.0;
  std::string par;
  
  Vector3 center;
  Material mat;
  
  while(pos != std::string::npos){
    
    par = line.substr(init, pos-init);
    init = pos + 1;
    pos = line.find(ch, init);
    
    if(centerCount < 4){ 
      if(centerCount < 3) center[centerCount] = stof(par);
      else radius = stof(par);
      centerCount++;
    }
    else {
      if(stoi(par) == LAMBERTIAN) mat = loadMaterial(line.substr(init,line.size()),pos,ch,LAMBERTIAN);
      else if(stoi(par) == METAL) mat = loadMaterial(line.substr(init,line.size()),pos,ch,METAL);
      else if(stoi(par) == DIELECTRIC) mat = loadMaterial(line.substr(init,line.size()),pos,ch,DIELECTRIC);
      else if(stoi(par) == DIFFUSE_LIGHT) mat = loadMaterial(line.substr(init,line.size()),pos,ch,DIFFUSE_LIGHT);
    }
  }
  return Sphere(center,radius,mat);
}

void Scene::loadScene(int loadType, const std::string &filename){
  
  std::cout << "Loading scene..." << std::endl;
  
  if(loadType == FFILE) sceneFromFile(filename);
  else if(loadType == RANDOM) sceneRandom();
  
  std::cout << "Scene loaded" << std::endl;
}

void Scene::sceneFromFile(const std::string &filename) {
  std::cout << "Scene file from " << filename << std::endl;
  
  std::ifstream file("Scenes/"+filename);
  std::ifstream aux("Scenes/"+filename);
  std::string line;
  
  int lines = 0;
  int objs = 0;
  
  if(file.fail() or aux.fail()) throw std::runtime_error("Something goes wrong");
  
  while(std::getline(aux,line)) ++lines;
  
  Sphere *list = new Sphere[lines];
  
  while(std::getline(file, line)) {
    if(line[0] == '1'){ 
      list[objs] = loadSphere(line.substr(2,line.size()));
      ++objs;
    }
  }
  
  spheres = new Sphere[size];
  std::copy(list, list + objs, spheres);
  
  size = objs;
  
}

void Scene::sceneRandom() {
  
  std::cout << "Random scene using dist = " << dist << std::endl;
  
  int n = (2*dist)*(2*dist)+5;
  
  std::cout << "Max size " << n << std::endl;
  
  Sphere *list = new Sphere[n];

  int objs = 0;
  
  list[objs] = Sphere(Vector3(0,-1000,0),1000, Material(LAMBERTIAN, Vector3(0.5,0.5,0.5)));
  objs++;
  
  for(int a = -dist; a < dist; a++){
    for(int b = -dist; b < dist; b++){
      float choose_mat = Random;
      Vector3 center(a+0.9*Random, 0.2, b+0.9*Random);
      if((center-Vector3(0,0,0)).length() > 0.995){
        if(choose_mat < 0.8){ //diffuse
            
          list[objs] = Sphere(center, 0.2, Material(LAMBERTIAN, Vector3(Random*Random, Random*Random, Random)));
        }
        else if(choose_mat < 0.90){ //metal
            
          list[objs] = Sphere(center, 0.2, Material(METAL, Vector3(0.5*(1+Random), 0.5*(1+Random), 0.5*(1+Random)), 0.5*Random));
        }
        else if(choose_mat < 0.95){
          list[objs] = Sphere(center, 0.2, Material(DIELECTRIC,Vector3::One(),-1.0,1.5));
        }
        else {
          list[objs] = Sphere(center, 0.2, Material(DIFFUSE_LIGHT, Vector3::One()));
        }
        objs++;
      }
    }
  }
    
  list[objs] = Sphere(Vector3(0,1,0), 1.0, Material(DIELECTRIC,Vector3::One(),-1,1.5));
  objs++;
  
  list[objs] = Sphere(Vector3(-4,1,0),1.0, Material(LAMBERTIAN,Vector3(0.4,0.2,0.1)));
  objs++;
  
  list[objs] = Sphere(Vector3(4,1,0),1.0, Material(METAL,Vector3(0.7,0.6,0.5),0.0));
  objs++;
  
  list[objs] = Sphere(Vector3(4,1,5), 1.0, Material(METAL,Vector3(0.9, 0.2, 0.2),0.0));
  objs++;
  
  spheres = new Sphere[objs];
  std::copy(list, list + objs, spheres);
  
  size = objs;
  
}

Sphere *Scene::getObjects() {
  return spheres;
}

unsigned int Scene::getSize() {
  return size;
}

