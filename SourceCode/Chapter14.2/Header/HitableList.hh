#ifndef _HITABLELIST_HH_INCLUDE
#define _HITABLELIST_HH_INCLUDE

#include "Hitable.hh"

class HitableList: public Hitable {
  
public:
    HitableList();
    HitableList(Hitable **l, int n);
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual bool bounding_box(float t0, float t1, aabb& box) const;
    
    Hitable **list;
    int list_size;
    
};

#endif /* _HITABLELIST_HH_INCLUDE */
