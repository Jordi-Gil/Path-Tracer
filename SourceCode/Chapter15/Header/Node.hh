#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Sphere.hh"
#include "Triangle.hh"
#include "Helper.hh"
#include <algorithm>

struct int2 {
    
  int x; int y;
  
  int2(int tx, int ty):x(tx), y(ty) {}
    
};

class Node {

public:
  Node();

  bool checkCollision(const Ray& r, float tmin, float tmax, hit_record& rec);
  
  Node *left;   // Left child
  Node *right;  // Right child
  Node *parent; // Parent

  //Sphere *obj; //Null if is an internal node, Object if is a Leaf
  Triangle *obj; //Null if is an internal node, Object if is a Leaf
  
  aabb box;
};


#endif /* Node_HH_INCLUDE */
