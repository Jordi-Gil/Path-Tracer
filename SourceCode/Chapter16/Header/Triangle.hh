#ifndef _TRIANGLE_HH_INCLUDE
#define _TRIANGLE_HH_INCLUDE

#include <stdexcept>

#include "Material.hh"
#include "aabb.hh"

class Triangle {
  
public:
    
  Triangle() {}
  Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 t = Vector3::One());
  bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  void bounding_box(aabb& box);
  
  Vector3 operator[](int i) const;
  Vector3& operator[](int i);
  Vector3 getCentroid();
  Material getMaterial();
  
  long long getMorton();
  void setMorton(long long code);
  aabb getBox();
  void resizeBoundingBox();
  
private:
  
  Vector3 vertex[3];
  Vector3 centroid;
  Material mat_ptr;
  Vector3 uv;
  long long morton_code;
  aabb box;
  
};

#endif /* _TRIANGLE_HH_INCLUDE */
