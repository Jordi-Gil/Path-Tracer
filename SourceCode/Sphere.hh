#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Hitable.hh"

class Sphere: public Hitable {
  
public:
    Sphere() {}
    Sphere(Vector3 cen, float r, Material *m): center(cen), radius(r), mat_ptr(m) {};
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record &rec) const;
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
    
};

#endif /* _SPHERE_HH_INCLUDE */
