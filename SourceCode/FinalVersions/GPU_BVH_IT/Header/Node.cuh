#ifndef Node_HH_INCLUDE
#define Node_HH_INCLUDE

#include "Triangle.cuh"
#include "Helper.cuh"

#define STACK_SIZE 1024

class Node{

public:

  __host__ __device__ Node();
  
  __device__ bool intersect(const Ray& r, float tmin, float tmax, hit_record& rec);

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
