#include "Node.cuh"

__host__ __device__ Node::Node() {
    obj = NULL;
    left = NULL;
    right = NULL;
    parent = NULL;
}

__device__ bool Node::intersect(const Ray& r, float tmin, float tmax, hit_record& rec) {
    
  float t = tmax;
  
  Node *stack[STACK_SIZE];
  int top = -1;
  
  stack[++top] = this;
  
  hit_record rec_left, rec_right;
  bool hit_left = false, hit_right = false;
  
  do {
  
    hit_record rec_aux;
    
    Node *new_node = stack[top--];
    
    if(new_node->box.hit(r, tmin, tmax)) {
      
      if(new_node->isLeaf) {
          bool hit = (new_node->obj)->hit(r, tmin, tmax, rec_aux);
          if(hit && rec_aux.t < t){
            
            t = rec_aux.t;
            
            if(new_node->isRight){
              rec_right = rec_aux;
              hit_right = true;
            }
            else if(new_node->isLeft){
              rec_left = rec_aux;
              hit_left = true;
            }
            
          }
        }
        
      else {
        
        stack[++top] = new_node->right;
        stack[++top] = new_node->left;
        
      }
    }
    
  } while(top != -1);
  
  if(hit_left && hit_right) {
    if(rec_left.t < rec_right.t) rec = rec_left;
    else rec = rec_right;
    return true;
  }
  else if(hit_left){ 
    rec = rec_left; 
    return true;
  }
  else if(hit_right) {
    rec = rec_right;
    return true;
  }
  else return false;
 
}

