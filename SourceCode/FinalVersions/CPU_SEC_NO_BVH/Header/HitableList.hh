#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Triangle.hh"

class HitableList {
  
public:
  
    HitableList();
    HitableList(Triangle *l, int n);
    bool intersect(const Ray& r, float t_min, float t_max, hit_record& rec);
    int length();
    Triangle *getObjects();

private:    
    
    Triangle *list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */
