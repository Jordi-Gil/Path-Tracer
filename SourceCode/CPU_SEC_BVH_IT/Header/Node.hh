#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Triangle.hh"
#include "Helper.hh"
#include <algorithm>

#define STACK_SIZE 1024

struct int2 {
    
  int x; int y;
  
  int2(int tx, int ty):x(tx), y(ty) {}
    
};

class Node {

public:
  Node();

  bool intersect(Node* root, const Ray& r, float tmin, float tmax, hit_record& rec);
  
  Node *left;   // Left child
  Node *right;  // Right child
  Node *parent; // Parent
  
  Triangle *obj;
  aabb box;
  bool isLeaf = false;
  bool isRight = false;
  bool isLeft = false;
  
};


#endif /* Node_HH_INCLUDE */
