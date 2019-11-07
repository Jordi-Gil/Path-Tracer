#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Material.hh"

class Sphere {
  
public:
    
  Sphere() {}
  Sphere(Vector3 cen, float r, Material mat);
  bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  Vector3 getCenter();
  float getRadius();
  Material getMaterial();
  
private:  
  
  Vector3 center;
  float radius;
  Material mat_ptr;
  
};

#endif /* _MOVING_SPHERE_HH_INCLUDE */
