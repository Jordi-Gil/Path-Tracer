#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "MovingSphere.cuh"
#include "Helper.cuh"
#include <cstring>

class Node{

public:

    __host__ __device__ Node();
  
    __host__ __device__ bool checkCollision(const Ray& r, float tmin, float tmax, hit_record& rec);

    Node *left;   // Left child
    Node *right;  // Right child
    Node *parent; // Parent
  
    MovingSphere *obj; //Null if is an internal node, Object if is a Leaf
  
    aabb box;
    
    int id;
    char *name;
  
};


#endif /* Node_HH_INCLUDE */
