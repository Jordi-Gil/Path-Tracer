#ifndef _OBJ_HH_INCLUDE
#define _OBJ_HH_INCLUDE

#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include "Triangle.hh"

enum fileTye {
  OBJF, TRIF
};

class Obj {
  
public:
    
  Obj() {}
  Obj(int type, const std::string &filename);
  void loadFromTXT(const std::string &filename);
  
  Triangle *getTriangles();
  

private:
  
  Triangle *triangles;
  unsigned int size;
  Vector3 center;
  
};

#endif /* _OBJ_HH_INCLUDE */
