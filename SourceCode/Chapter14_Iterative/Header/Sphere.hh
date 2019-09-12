#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Helper.hh"
#include "Hitable.hh"
#include "Material.hh"

class Sphere: public Hitable {
  
public:
    
    Sphere() {}
    Sphere(Vector3 cen, float r, Material *mat);
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual bool bounding_box(float t0, float t1, aabb& box) const;
    
    int get_id(){return id;}
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
    int id;
    unsigned int morton_code;
};

#endif /* _SPHERE_HH_INCLUDE */
