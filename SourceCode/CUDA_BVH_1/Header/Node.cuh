#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Hitable.cuh"
#include "Helper.cuh"

class Node{

public:

    __device__ Node();
  
    __device__ bool checkCollision(const Ray& r, float tmin, float tmax, hit_record& rec);

    Node *left;   // Left child
    Node *right;  // Right child
    Node *parent; // Parent
  
    Hitable *obj; //Null if is an internal node, Object if is a Leaf
  
    aabb box;
    
    char *name;
  
};


#endif /* Node_HH_INCLUDE */
