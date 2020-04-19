#include "Node.cuh"

__host__ __device__ Node::Node() {
    obj = NULL;
    left = NULL;
    right = NULL;
    parent = NULL;
}

__device__ bool Node::intersect(const Ray& r, float tmin, float tmax, hit_record& rec) {
    
  if(box.hit(r, tmin, tmax)) {
    hit_record left_rec, right_rec;

    bool hit_left;
    bool hit_right;

    if(left->obj) hit_left = (left->obj)->hit(r, tmin, tmax, left_rec);
    else hit_left = left->intersect(r, tmin, tmax, left_rec);
    
    if(right->obj) hit_right = (right->obj)->hit(r, tmin, tmax, right_rec);
    else hit_right = right->intersect(r, tmin, tmax, right_rec);
    
    if(hit_left && hit_right) {
    
      if(left_rec.t < right_rec.t) rec = left_rec;
      else rec = right_rec;
      
      return true;
    }
    else if (hit_left) {
      rec = left_rec;
      return true;
    }
    else if(hit_right) {
      rec = right_rec;
      return true;
    }
    else return false;
  }
  else return false;
}

