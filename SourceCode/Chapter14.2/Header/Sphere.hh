#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Hitable.hh"
#include "Material.hh"

class Sphere: public Hitable {
  
public:
    Sphere() {}
    Sphere(Vector3 cen, float r, Material *mat): center(cen), radius(r), mat_ptr(mat){};
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual bool bounding_box(float t0, float t1, aabb& box) const;
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
};

#endif /* _SPHERE_HH_INCLUDE */
