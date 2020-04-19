#ifndef _OBJ_HH_INCLUDE
#define _OBJ_HH_INCLUDE

#include <algorithm>
#include <limits>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "Triangle.cuh"
#include "Mat4.cuh"

#define INF std::numeric_limits<float>::infinity()
#define MIN std::numeric_limits<float>::lowest()

enum fileTye {
  OBJF, TRIF
};

class Obj {
  
public:
    
  __host__  Obj() {}
  __host__  Obj(int type, const std::string &filename, bool matB = false, Material m = Material(), int textureIndex = -1);
  __host__  void loadFromTXT(const std::string &filename);
  __host__  void loadFromObj(const std::string &filename);
  
  __host__  Triangle *getTriangles();
  __host__  Vector3 getMax();
  __host__  Vector3 getMin();
  __host__  Vector3 getObjCenter();
  __host__  int getSize();
  
  __host__  void applyGeometricTransform(mat4 m);
  
private:
  
  Triangle *triangles;
  int size;
  Vector3 center;
  Vector3 max;
  Vector3 min;
  bool materialB;
  Material material;
  
};

__host__ __device__ inline void compare(Vector3 &max, Vector3 &min, Vector3 point) {
  
  if(point[0] > max[0]) max[0] = point[0]; //x
  if(point[1] > max[1]) max[1] = point[1]; //y
  if(point[2] > max[2]) max[2] = point[2]; //z
  
  if(point[0] < min[0]) min[0] = point[0]; //x
  if(point[1] < min[1]) min[1] = point[1]; //y
  if(point[2] < min[2]) min[2] = point[2]; //z
}




#endif /* _OBJ_HH_INCLUDE */
