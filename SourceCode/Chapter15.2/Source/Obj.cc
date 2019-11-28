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
      triangles = new Shape[size];
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
        triangles[count].tag = TRIANGLE;
        Triangle tri = Triangle(filename+std::to_string(count),tri[0],tri[1],tri[2],(materialB) ? material : mat);
        triangles[count].t = tri;
        compare(max_l, min_l, tri[0]); compare(max_l, min_l, tri[1]); compare(max_l, min_l, tri[2]);
        ++count;
      }
    }
  }
  max = max_l;
  min = min_l;
  center = Vector3((max_l[0] + min_l[0])/2, (max_l[1] + min_l[1])/2, (max_l[2] + min_l[2])/2);
}

void Obj::loadFromObj(const std::string &filename) {
  
  std::ifstream file("Models/"+filename);
  if(file.fail()) throw std::runtime_error("Something goes wrong");
  
  std::string line;
  
  Vector3 max_l(MIN);
  Vector3 min_l(INF);
  
  Material mat = (materialB) ? material : Material(LAMBERTIAN, Vector3(0.4));
  
  std::vector<Vector3> vertexs;
  std::vector<Vector3> coordText;
  std::vector<Shape> aux;
  int count = 0;
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
        Shape sp;
        sp.tag = TRIANGLE;
        Triangle tri = Triangle(filename+std::to_string(count),vertexs[face[0][0]-1],vertexs[face[i][0]-1],vertexs[face[i+1][0]-1],mat);
        sp.t = tri;
        aux.push_back(sp);
        count++;
      }
    }
  }
  
  size = aux.size();
  triangles = new Shape[size];
  std::copy(aux.begin(), aux.end(), triangles);
  
  max = max_l;
  min = min_l;
  center = Vector3((max_l[0] + min_l[0])/2, (max_l[1] + min_l[1])/2, (max_l[2] + min_l[2])/2);
  
}

Obj::Obj(int type, const std::string &filename, bool mat, Material m) {
  
  std::cout << "Loading OBJ " << filename << std::endl;
  
  materialB = mat;
  if(materialB) material = m;
  
  if(type == TRIF) loadFromTXT(filename);
  else if(type == OBJF) loadFromObj(filename);
  else throw std::runtime_error("Invalid file type");
  
  std::cout << filename << " Center: " << center << std::endl;
}
  
Shape *Obj::getTriangles(){
  return triangles;
}

Vector3 Obj::getMax() {
  return max;
}

Vector3 Obj::getMin() {
  return min;
}

Vector3 Obj::getObjCenter() {
  return center;
}

int Obj::getSize() {
  return size;
}

void Obj::applyGeometricTransform(mat4 m) {
  
  for(int i = 0; i < size; i++){
    for(int j = 0; j < 3; j++){
      triangles[i].t[j] = Vector3( 
              m(0,0)*triangles[i].t[j][0] + m(0,1)*triangles[i].t[j][1] + m(0,2)*triangles[i].t[j][2] + m(0,3)*1,
              m(1,0)*triangles[i].t[j][0] + m(1,1)*triangles[i].t[j][1] + m(1,2)*triangles[i].t[j][2] + m(1,3)*1,
              m(2,0)*triangles[i].t[j][0] + m(2,1)*triangles[i].t[j][1] + m(2,2)*triangles[i].t[j][2] + m(2,3)*1
                               );
    }
    triangles[i].t.resizeBoundingBox();
  }
}
