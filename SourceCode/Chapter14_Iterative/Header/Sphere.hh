#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Helper.hh"
#include "Hitable.hh"
#include "Material.hh"
#include "aabb.hh"

class Sphere: public Hitable {
  
public:
    
    Sphere() {}
    Sphere(Vector3 cen, float r, Material *mat);
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual void bounding_box(float t0, float t1, aabb& box) const;
    virtual unsigned int getMorton() const;
    virtual void setMorton(unsigned int code);
    virtual aabb getBox() const;
    virtual Vector3 getCenter() const;
    
    Vector3 center;
    float radius;
    Material *mat_ptr;
    unsigned int morton_code;
    aabb box;
};

#endif /* _SPHERE_HH_INCLUDE */
