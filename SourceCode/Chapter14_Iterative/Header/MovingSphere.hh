#ifndef _MOVING_SPHERE_HH_INCLUDE
#define _MOVING_SPHERE_HH_INCLUDE

#include "Helper.hh"
#include "Hitable.hh"
#include "Material.hh"

class MovingSphere: public Hitable {
  
public:
    
    MovingSphere() {}
    MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat);
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual bool bounding_box(float t0, float t1, aabb& box) const;
    
    Vector3 center(float time) const;
    
    unsigned int get_morton_code(){return morton_code_0;}
    
private:    
    
    Vector3 center0, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
    unsigned int morton_code_0;
    unsigned int morton_code_1;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
