#include "BVH_node.hh"

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

BVH_node::BVH_node(Hitable **list, int n, float time0, float time1) {

    int axis = int(3*(rand()/(RAND_MAX + 1.0)));
  
    if (axis == 0) qsort(list, n, sizeof(Hitable *), box_x_compare);
    else if (axis == 1) qsort(list, n, sizeof(Hitable *), box_y_compare);
    else qsort(list, n, sizeof(Hitable *), box_z_compare);
  
    if(n == 1) left = right = list[0];
    else if (n == 2){
        left = list[0];
        right = list[1];
    }
    else {
        left = new BVH_node(list, n/2, time0, time1);
        right = new BVH_node(list + n/2, n - n/2, time0, time1); 
    }
  
    aabb box_left, box_right;
    if(!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right)) 
        std::cerr << "No bouding box in BVH_node constructor" << std::endl;
  
    box = surrounding_box(box_left, box_right);
}

bool BVH_node::hit(const Ray& r, float tmin, float tmax, hit_record& rec) const {

    if(box.hit(r, tmin, tmax)) {
        
        hit_record left_rec, right_rec;
        
        bool hit_left = left->hit(r, tmin, tmax, left_rec);     //If BVH_NODE -> Internal Node / If Sphere/MovingSphere -> LeafNode
        bool hit_right = right->hit(r, tmin, tmax, right_rec);  //If BVH_NODE -> Internal Node / If Sphere/MovingSphere -> LeafNode
    
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

bool BVH_node::bounding_box(float t0, float t1, aabb& b) const {

    t0 = t0;
    t1 = t1;
    b = box;
    return true;
  
}
