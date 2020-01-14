#include "Obj.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__host__ Material loadMat(const std::string &line, int type, int texType) {

  std::stringstream ssin(line);  
  std::string par;
  
  int albedoCount = 0;
  float fuzz = -1.0;
  float ref_idx = -1.0;
  bool loaded = false;
  Vector3 albedo;
  std::string imageFilename;
  int nx, ny, nc;
  unsigned char *image;
  
  while(ssin >> par and !loaded) {
  
    if(albedoCount < 3 and texType == CONSTANT){
      albedo[albedoCount] = stof(par);
      albedoCount++;
    } else {
      if(texType == IMAGE){
        imageFilename = "Textures/"+par;
        loaded = true;
      }
      if(type == METAL) { fuzz = stof(par); loaded = true; }
      else if(type == DIELECTRIC) { ref_idx = stof(par); loaded = true; }
      else if(albedoCount == 3) loaded = true;
    }
  }
  
  if(texType == IMAGE) image = stbi_load(imageFilename.c_str(), &nx, &ny, &nc, 0);
  
  if(texType == CONSTANT) return Material(type,Texture(texType, albedo), fuzz,ref_idx);
  
  return Material(type,Texture(texType, Vector3::Zero(), image, nx, ny), fuzz,ref_idx);
}

__host__  Vector3 loadVector3(const std::string &line){
  
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

__host__  void Obj::loadFromTXT(const std::string &filename) {
  
  std::ifstream file("Models/"+filename);
  if(file.fail()) throw std::runtime_error("Something goes wrong");
  
  std::string line;
  
  Vector3 max_l(MIN);
  Vector3 min_l(INF);
  
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
        std::string par2;
        ssin >> par2;
        int type = (par2 == "CONSTANT") ? CONSTANT : IMAGE;
        if(par == "L") mat = loadMat(line.substr(line.find(par2)+par2.size()),LAMBERTIAN, type);
        else if(par == "M") mat = loadMat(line.substr(line.find(par2)+par2.size()),METAL, type);
        else if(par == "D") mat = loadMat(line.substr(line.find(par2)+par2.size()),DIELECTRIC, type);
        else if(par == "DL") mat = loadMat(line.substr(line.find(par2)+par2.size()),DIFFUSE_LIGHT, type);
        
        cTri = 0;
        triangles[count] = Triangle(tri[0],tri[1],tri[2],(materialB) ? material : mat);
        compare(max_l, min_l, tri[0]); compare(max_l, min_l, tri[1]); compare(max_l, min_l, tri[2]);
        ++count;
      }
    }
  }
  max = max_l;
  min = min_l;
  center = Vector3((max_l[0] + min_l[0])/2, (max_l[1] + min_l[1])/2, (max_l[2] + min_l[2])/2);
}

__host__  void Obj::loadFromObj(const std::string &filename) {
  
  std::ifstream file("Models/"+filename);
  if(file.fail()) throw std::runtime_error("Something goes wrong");
  
  std::string line;
  
  Vector3 max_l(MIN);
  Vector3 min_l(INF);
  
  Material mat = (materialB) ? material : Material(LAMBERTIAN, Texture(CONSTANT, Vector3(0.4)));
  
  
  std::vector<Vector3> vertexs;
  std::vector<Vector3> coordText;
  std::vector<Triangle> aux;
  
  while(std::getline(file,line)){
    
    std::istringstream ssin(line);
    std::string par;
    
    ssin >> par;
    
    if(par == "v") {
      float x, y, z;
      
      ssin >> x >> y >> z;
      Vector3 v = Vector3(x,y,z);
      vertexs.push_back(v);
      compare(max_l, min_l, v);
      
    }
    else if(par == "vt") {
      float u, v;
      ssin >> u >> v;
      coordText.push_back(Vector3(u,v,-1));
      std::cout << coordText[coordText.size()-1] << std::endl;
    }
    else if(par == "f") {
      
      std::vector<Vector3> face;
      std::string idxStr;
      while(ssin >> idxStr) {
        std::istringstream idx(idxStr);
        std::string v_str, vt_str, vn_str;
        std::getline(idx, v_str, '/');
        std::getline(idx, vt_str, '/');
        std::getline(idx, vn_str, '/');
        
        int v = (v_str != "") ? stoi(v_str) : -1;
        int vt = (vt_str != "") ? stoi(vt_str) : -1;
        int vn = (vn_str != "") ? stoi(vn_str) : -1;
        
        face.push_back(Vector3(v,vt,vn));
      }
      
      for(int i = 1; i < face.size()-1; i++){
        
        aux.push_back(Triangle(vertexs[face[0][0]-1],vertexs[face[i][0]-1],vertexs[face[i+1][0]-1],mat,coordText[face[0][1]-1],coordText[face[i][1]-1],coordText[face[i+1][1]-1]));
      }
    }
  }
  
  size = aux.size();
  triangles = new Triangle[size];
  std::copy(aux.begin(), aux.end(), triangles);
  
  max = max_l;
  min = min_l;
  center = Vector3((max_l[0] + min_l[0])/2, (max_l[1] + min_l[1])/2, (max_l[2] + min_l[2])/2);
  
}

__host__  Obj::Obj(int type, const std::string &filename, bool mat, Material m) {
  
  materialB = mat;
  if(materialB) material = m;
  
  if(type == TRIF) loadFromTXT(filename);
  else if(type == OBJF) loadFromObj(filename);
  else throw std::runtime_error("Invalid file type");
  
  std::cout << filename << " Center: " << center << std::endl;
}
  
__host__  Triangle *Obj::getTriangles(){
  return triangles;
}

__host__  Vector3 Obj::getMax() {
  return max;
}

__host__ Vector3 Obj::getMin() {
  return min;
}

__host__  Vector3 Obj::getObjCenter() {
  return center;
}

__host__  int Obj::getSize() {
  return size;
}

__host__  void Obj::applyGeometricTransform(mat4 m) {
  
  for(int i = 0; i < size; i++){
    for(int j = 0; j < 3; j++){
      triangles[i][j] = Vector3( 
              m(0,0)*triangles[i][j][0] + m(0,1)*triangles[i][j][1] + m(0,2)*triangles[i][j][2] + m(0,3)*1,
              m(1,0)*triangles[i][j][0] + m(1,1)*triangles[i][j][1] + m(1,2)*triangles[i][j][2] + m(1,3)*1,
              m(2,0)*triangles[i][j][0] + m(2,1)*triangles[i][j][1] + m(2,2)*triangles[i][j][2] + m(2,3)*1
                               );
    }
  }
}
