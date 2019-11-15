#include "Obj.hh"

Material loadMat(const std::string &line,int type) {

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


Vector3 loadVector3(const std::string &line){
  
  Vector3 v;
  std::stringstream ssin(line);
  std::string par;
  int count = 0;
  
  while(ssin >> par) {
    v[count] = stof(par);
    ++count;
  }
  return v;
}

void Obj::loadFromTXT(const std::string &filename) {
  
  std::ifstream file("Models/"+filename);
  if(file.fail()) throw std::runtime_error("Something goes wrong");
  
  std::string line;
  
  int cTri = 0;
  int count = 0;
  
  Vector3 *tri = new Vector3[3];
  
  while(std::getline(file, line)) {
    std::stringstream ssin(line);
    std::string par;
    
    ssin >> par;
    
    if(par[0] >= '0' and par[1] <= '9') {
      size = stoi(line);
      triangles = new Triangle[size];
    }
    else {
      if(par == "v\\" and cTri < 3) {
        tri[cTri] = loadVector3(line.substr(line.find(par)+par.size()));
        ++cTri;
      }
      if(par == "m\\" and cTri == 3) {
        ssin >> par;
        Material mat;
        
        if(par == "L") mat = loadMat(line.substr(line.find(par)+par.size()),LAMBERTIAN);
        else if(par == "M") mat = loadMat(line.substr(line.find(par)+par.size()),METAL);
        else if(par == "D") mat = loadMat(line.substr(line.find(par)+par.size()),DIELECTRIC);
        else if(par == "DL") mat = loadMat(line.substr(line.find(par)+par.size()),DIFFUSE_LIGHT);
        
        cTri = 0;
        triangles[count] = Triangle(tri[0],tri[1],tri[2],mat);
        ++count;
      }
    }
  }
  std::cout << "Count: " << count << std::endl;
  
  for(int i = 0; i < count; i++) {
    for(int j = 0; j < 3; j++) {
      if(j == 0) std::cout << "Triangle ";
      std::cout << triangles[i].getVertex(j) << " ";
      if(j == 2) std::cout << std::endl;
    }
  }
}

Obj::Obj(int type, const std::string &filename) {
  
  if(type == TRIF){
    std::cout << "TRIF file"<<std::endl;
    loadFromTXT(filename);
  }
}
  
Triangle *Obj::getTriangles(){
  return triangles;
}
