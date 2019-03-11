#ifndef _HITABLE_HH_INCLUDE
#define _HITABLE_HH_INCLUDE

#include "Ray.hh"

class Material;

struct hit_record{
    float t;
    Vector3 point;
    Vector3 normal;
    Material *mat_ptr;
};

class Hitable {
  
public:
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record &rec) const = 0;
    
};

#endif /* _HITABLE_HH_INCLUDE */
