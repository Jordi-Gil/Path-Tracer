#ifndef _OBJ_HH_INCLUDE
#define _OBJ_HH_INCLUDE

#include <algorithm>
#include <limits>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Triangle.hh"
#include "Mat4.hh"

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::lowest()

enum fileTye {
  OBJF, TRIF
};

class Obj {
  
public:
    
  Obj() {}
  Obj(int type, const std::string &filename, bool matB = false, Material m = Material());
  void loadFromTXT(const std::string &filename);
  void loadFromObj(const std::string &filename);
  
  Triangle *getTriangles();
  Vector3 getMax();
  Vector3 getMin();
  Vector3 getObjCenter();
  int getSize();
  
  void applyGeometricTransform(mat4 m);
  
private:
  
  Triangle *triangles;
  int size;
  Vector3 center;
  Vector3 max;
  Vector3 min;
  bool materialB;
  Material material;
};

inline void compare(Vector3 &max, Vector3 &min, Vector3 point) {
  
  if(point[0] > max[0]) max[0] = point[0]; //x
  if(point[1] > max[1]) max[1] = point[1]; //y
  if(point[2] > max[2]) max[2] = point[2]; //z
  
  if(point[0] < min[0]) min[0] = point[0]; //x
  if(point[1] < min[1]) min[1] = point[1]; //y
  if(point[2] < min[2]) min[2] = point[2]; //z
}




#endif /* _OBJ_HH_INCLUDE */
