#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Hitable.hh"
#include "Helper.hh"
#include <algorithm>

typedef std::vector< Hitable* > vh;

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
  
    Hitable *obj; //Null if is an internal node, Object if is a Leaf
  
    aabb box;
    
    std::string name;
  
};

class BVH {
    BVH() {}
    
    Node *internal;
    Node *leaves;
    
    int numSpheres;
}


#endif /* Node_HH_INCLUDE */
