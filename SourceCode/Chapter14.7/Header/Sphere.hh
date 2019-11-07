#ifndef _SPHERE_HH_INCLUDE
#define _SPHERE_HH_INCLUDE

#include "Material.hh"
#include "aabb.hh"

class Sphere {
  
public:
    
  Sphere() {}
  Sphere(Vector3 cen, float r, Material mat);
  bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  void bounding_box(aabb& box);
  void setMorton(long long code);
  
  Vector3 getCenter();
  float getRadius();
  Material getMaterial();
  long long getMorton();
  aabb getBox();
  

private:
  
  Vector3 center;
  float radius;
  Material mat_ptr;
  long long morton_code;
  aabb box;
};

struct ObjEval{
    
  inline bool operator()(Sphere a, Sphere b){
    return (a.getMorton() < b.getMorton());
  }

};

#endif /* _SPHERE_HH_INCLUDE */