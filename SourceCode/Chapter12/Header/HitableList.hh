#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Sphere.hh"

class HitableList {
  
public:
  
    HitableList();
    HitableList(Sphere *l, int n);
    bool checkCollision(const Ray& r, float t_min, float t_max, hit_record& rec);
    int length();
    Sphere *getObjects();

private:    
    
    Sphere *list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */
