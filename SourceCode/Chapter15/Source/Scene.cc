#include "Scene.hh"

void compare(Vector3 &max, Vector3 &min, Vector3 point) {
    
  if(point[0] > max[0]) max[0] = point[0]; //x
  if(point[1] > max[1]) max[1] = point[1]; //y
  if(point[2] > max[2]) max[2] = point[2]; //z
  
  if(point[0] < min[0]) min[0] = point[0]; //x
  if(point[1] < min[1]) min[1] = point[1]; //y
  if(point[2] < min[2]) min[2] = point[2]; //z
}

Material loadMaterial(const std::string &line, size_t &pos, char ch,int type) {

	std::cout << "Type " << type << std::endl;
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

Triangle loadTriangle(const std::string &line) {
	
	std::cout << "Line: " << line << std::endl;
	
	char ch = ' ';
	size_t pos = line.find(ch);
	size_t init = 0;
	int vertexCount = 0;
	int trCount = 0;
	std::string par;
	
	Vector3 *position = new Vector3[3];
	Material mat;
	
	while(pos != std::string::npos) {
		par  = line.substr(init, pos-init);
		init  = pos + 1;
		pos = line.find(ch, init);
		std::cout << "par: " << par << std::endl;
		if(trCount < 3) {
			if(vertexCount < 3) {
				std::cout << "Triangle: " << trCount << " Vertex: " << vertexCount << " Value: " << par << std::endl;
				position[trCount][vertexCount] = stof(par);
				++vertexCount;
			}
			else{
				std::cout << "Vertex " << trCount << " : " << position[trCount] << std::endl;
				vertexCount = 0;
				++trCount;
				if(trCount < 3) {
					std::cout << "Triangle: " << trCount << " Vertex: " << vertexCount << " Value: " << par << std::endl;
					position[trCount][vertexCount] = stof(par);
					++vertexCount;
				}
			}
		}
		else {
      if(stoi(par) == LAMBERTIAN) {
				std::cout << "Lambertian ";
				mat = loadMaterial(line.substr(init,line.size()),pos,ch,LAMBERTIAN);
			}
      else if(stoi(par) == METAL) mat = loadMaterial(line.substr(init,line.size()),pos,ch,METAL);
      else if(stoi(par) == DIELECTRIC) mat = loadMaterial(line.substr(init,line.size()),pos,ch,DIELECTRIC);
      else if(stoi(par) == DIFFUSE_LIGHT) mat = loadMaterial(line.substr(init,line.size()),pos,ch,DIFFUSE_LIGHT);
		}
	}
	std::cout << "Material " << mat.getName() << " " << mat.getAlbedo() << std::endl;
	return Triangle(position[0],position[1],position[2], mat);
}

void Scene::loadScene(int loadType, const std::string &filename){
  
  std::cout << "Loading scene..." << loadType << std::endl;
  
  if(loadType == FFILE) sceneFromFile(filename);
  else if(loadType == RANDOM) sceneRandom();
  else if(loadType == TRIANGL) sceneTriangle();
  
  std::cout << "Scene loaded" << std::endl;
}

void Scene::sceneFromFile(const std::string &filename) {
  std::cout << "Scene file from " << filename << std::endl;
  
  std::ifstream file("Scenes/"+filename);
  std::ifstream aux("Scenes/"+filename);
  std::string line;
  
  int lines = 0;
  int objs_sp = 0;
	int objs_tr = 0;
  
  if(file.fail() or aux.fail()) throw std::runtime_error("Something goes wrong");
  
  while(std::getline(aux,line)) ++lines;
  
  Sphere *list_1 = new Sphere[lines];
	Triangle *list_2 = new Triangle[lines];
  
  Vector3 max(MIN);
  Vector3 min(INF);
  
  while(std::getline(file, line)) {
    if(line[0] == '1'){ 
      list_1[objs_sp] = loadSphere(line.substr(2,line.size()));
      compare(max, min, list_1[objs_sp].getCenter());
      ++objs_sp;
    }
    else if(line[0] == '2') {
      list_2[objs_tr] = loadTriangle(line.substr(2,line.size()));
			compare(max,min, list_2[objs_tr].getCentroid());
			++objs_tr;
    }
  }
  
  std::cout << "Real size " << objs_sp+objs_tr << std::endl;
  std::cout << "Max " <<  max << " Min " << min << std::endl;
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
  for(int idx = 0; idx < objs_sp; idx++) {
      
    Vector3 point = list_1[idx].getCenter();
    
    point[0] = ((point[0] - min_x)/(max_x - min_x));
    point[1] = ((point[1] - min_y)/(max_y - min_y));
    point[2] = ((point[2] - min_z)/(max_z - min_z));
    
    list_1[idx].setMorton(Helper::morton3D(point[0],point[1],point[2]));
  }

    
  max_x = max[0]; max_y = max[1]; max_z = max[2];
  min_x = min[0]; min_y = min[1]; min_z = min[2];
  
	for(int idx = 0; idx < objs_tr; idx++) {
		
		Vector3 point = list_2[idx].getCentroid();
		std::cout << "BOX triangle " << idx << " " << list_2[idx].getBox().min() << " " << list_2[idx].getBox().max() << std::endl;
		std::cout << "Centroid: " << point << std::endl;
		
		point[0] = ((point[0] - min_x)/(max_x - min_x));
		point[1] = ((point[1] - min_y)/(max_y - min_y));
		point[2] = ((point[2] - min_z)/(max_z - min_z));

		list_2[idx].setMorton(Helper::morton3D(point[0],point[1],point[2]));

  }  
  
  spheres = new Sphere[objs_sp];
  std::copy(list_1, list_1 + objs_sp, spheres);
  std::sort(spheres, spheres + objs_sp, ObjEval());
	
	triangles = new Triangle[objs_tr];
  std::copy(list_2, list_2 + objs_tr, triangles);
  std::sort(triangles, triangles + objs_tr, TriangleEval());
	
  size = objs_tr;
  
}

void Scene::sceneRandom() {
  
  std::cout << "Random scene using dist = " << dist << std::endl;
  
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
  std::sort(spheres, spheres + objs, ObjEval());
  
  
  size = objs;
  
}

void Scene::sceneTriangle() {
  
  std::cout << "Triangle scene" << std::endl;
  
  triangles = new Triangle[2];
	
	Vector3 max(MIN);
  Vector3 min(INF);
	
	int objs = 0;
  
  triangles[objs] = Triangle(Vector3(-4.0,0.0,0.0), Vector3(-2.0,0.0,0.5), Vector3(-3.0,-1.0,0.0) ,Material(LAMBERTIAN, Vector3(1.0,0.0,0.0)));
	compare(max, min, triangles[0].getCentroid()); objs++;
	
	triangles[objs] = Triangle(Vector3(0.0,0.0,0.0), Vector3(2.0,0.0,0.0), Vector3(1.0,-1.0,0.0) ,Material(METAL, Vector3(1.0,0.0,0.0)));
  compare(max, min, triangles[1].getCentroid()); objs++;
  
  float max_x = max[0]; float max_y = max[1]; float max_z = max[2];
  float min_x = min[0]; float min_y = min[1]; float min_z = min[2];
  
    for(int idx = 0; idx < objs; idx++) {
      
			Vector3 point = triangles[idx].getCentroid();
			std::cout << "BOX triangle " << idx << " " << triangles[idx].getBox().min() << " " << triangles[idx].getBox().max() << std::endl;
			std::cout << "Centroid: " << point << std::endl;
			
			point[0] = ((point[0] - min_x)/(max_x - min_x));
			point[1] = ((point[1] - min_y)/(max_y - min_y));
			point[2] = ((point[2] - min_z)/(max_z - min_z));
    
			triangles[idx].setMorton(Helper::morton3D(point[0],point[1],point[2]));
			
			std::cout << point << std::endl;
  
			triangles[idx].setMorton(Helper::morton3D(point[0],point[1],point[2])+1);
			
  }
  
  std::sort(triangles, triangles+objs, TriangleEval());
  size = 2;
}

Sphere *Scene::getObjects() {
  return spheres;
}

Triangle *Scene::getTriangles() {
  return triangles;
}

unsigned int Scene::getSize() {
  return size;
}

