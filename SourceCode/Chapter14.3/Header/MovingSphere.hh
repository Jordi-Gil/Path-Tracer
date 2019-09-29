#ifndef _MOVING_SPHERE_HH_INCLUDE
#define _MOVING_SPHERE_HH_INCLUDE

#include "Helper.hh"
#include "Hitable.hh"
#include "Material.hh"
#include "aabb.hh"

class MovingSphere: public Hitable {
  
public:
    
    MovingSphere() {}
    MovingSphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat);
    
    virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const;
    virtual void bounding_box(float t0, float t1, aabb& box) const;
    virtual unsigned int getMorton() const;
    virtual void setMorton(unsigned int code);
    virtual aabb getBox() const;
    virtual Vector3 getCenter() const;
    
    Vector3 get_center(float time) const;
    
    Vector3 center, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
    long long morton_code;
    aabb box;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
