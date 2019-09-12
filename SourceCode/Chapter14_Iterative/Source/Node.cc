#include "Node.hh"


int box_x_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) 
    std::cerr << "No bouding box in BVH_node constructor" << std::endl;
  if(box_left.min().x() - box_right.min().x() < 0.0) return -1;
  else return 1;
  
}

int box_y_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) 
    std::cerr << "No bouding box in BVH_node constructor" << std::endl;
  if(box_left.min().y() - box_right.min().y() < 0.0) return -1;
  else return 1;
  
}

int box_z_compare (const void *a, const void *b) {

  aabb box_left, box_right;
  
  Hitable *ah = *(Hitable**)a;
  Hitable *bh = *(Hitable**)b;
  
  if(!ah->bounding_box(0,0,box_left) || !bh->bounding_box(0,0, box_right)) 
    std::cerr << "No bouding box in BVH_node constructor" << std::endl;
  if(box_left.min().z() - box_right.min().z() < 0.0) return -1;
  else return 1;
  
}

Node::Node() {
    obj = NULL;
    left = NULL;
    right = NULL;
    parent = NULL;
}

bool Node::checkCollision(const Ray& r, float tmin, float tmax, hit_record& rec) {

    if(box.hit(r, tmin, tmax)) {
    
        hit_record left_rec, right_rec;
    
        bool hit_left;
        bool hit_right;
    
        if(left->obj) hit_left = (left->obj)->hit(r, tmin, tmax, left_rec);
        else hit_left = left->checkCollision(r, tmin, tmax, left_rec);
        
        if(right->obj) hit_right = (right->obj)->hit(r, tmin, tmax, right_rec);
        else hit_right = right->checkCollision(r, tmin, tmax, left_rec);
        
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

/*
bool Node::bounding_box(float t0, float t1, aabb& b) const {

  t0 = t0;
  t1 = t1;
  b = box;
  return true;
  
}
*/
