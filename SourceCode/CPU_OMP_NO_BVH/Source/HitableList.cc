#include "HitableList.hh"

HitableList::HitableList(){}

HitableList::HitableList(Triangle *l, int n){
    list = l;
    list_size = n;
}

bool HitableList::intersect(const Ray& r, float t_min, float t_max, hit_record &rec) {
    
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for(int i = 0; i < list_size; i++){
        if(list[i].hit(r, t_min, closest_so_far, temp_rec)){
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
    
}

int HitableList::length() {
  
  return list_size;
  
}

Triangle *HitableList::getObjects() {
  return list;
}
