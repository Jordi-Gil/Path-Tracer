#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

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

  bool intersect(Node *root, const Ray& cur_ray, float t_min, float t_max, hit_record& rec);
  
  Node *left;   // Left child
  Node *right;  // Right child
  Node *parent; // Parent

  //Sphere *obj; //Null if is an internal node, Object if is a Leaf
  Triangle *obj; //Null if is an internal node, Object if is a Leaf
  aabb box;
  
  bool isLeaf = false;
  bool isLeft = false;
  bool isRight = false;
  
};


#endif /* Node_HH_INCLUDE */
