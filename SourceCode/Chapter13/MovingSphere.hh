#ifndef _MOVING_SPHERE_HH_INCLUDE
#define _MOVING_SPHERE_HH_INCLUDE

#include "Hitable.hh"
#include "Material.hh"

class MovingSphere: public Hitable {
  
public:
    MovingSphere() {}
    MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat): center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(mat){};
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    
    Vector3 center(float time) const;
    
    Vector3 center0, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
