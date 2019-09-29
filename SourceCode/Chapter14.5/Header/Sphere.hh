#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Helper.hh"
#include "Hitable.hh"
#include "Material.hh"
#include "aabb.hh"

class Sphere {
  
public:
    
    Sphere() {}
    Sphere(Vector3 cen0, Vector3 cen1, float t0, float t1, float r, Material *mat);
    
    bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
    void bounding_box(float t0, float t1, aabb& box);
    
    Vector3 get_center(float time);
    
    Vector3 center, center1;
    float time0, time1;
    float radius;
    Material *mat_ptr;
    unsigned long long morton_code;
    aabb box;
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
