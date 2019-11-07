#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Sphere.cuh"
#include "Helper.cuh"

class Node{

public:

    __host__ __device__ Node();
  
    __device__ bool checkCollision(const Ray& r, float tmin, float tmax, hit_record& rec);

    Node *left;   // Left child
    Node *right;  // Right child
    Node *parent; // Parent
  
    Sphere *obj; //Null if is an internal node, Object if is a Leaf
  
    aabb box;
  
};


#endif /* Node_HH_INCLUDE */
