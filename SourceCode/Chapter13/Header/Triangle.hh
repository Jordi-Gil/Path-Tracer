#ifndef _TRIANGLE_HH_INCLUDE
#define _TRIANGLE_HH_INCLUDE

#include <stdexcept>

#include "Material.hh"

class Triangle {
  
public:
    
  Triangle() {}
  Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 uv1 = Vector3::One(), Vector3 uv3 = Vector3::One(), Vector3 uv2 = Vector3::One());
  bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  Vector3 operator[](int i) const;
  Vector3& operator[](int i);
  Vector3 getCentroid();
  Material getMaterial();
  
  
private:
  
  Vector3 vertex[3];
  Vector3 centroid;
  Material mat_ptr;
  Vector3 uv[3];
  
};

#endif /* _TRIANGLE_HH_INCLUDE */
