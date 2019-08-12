#ifndef BVH_node_HH_INCLUDE
#define BVH_node_HH_INCLUDE

#include "Hitable.hh"

class BVH_node: public Hitable {
public:
  BVH_node() {}
  BVH_node(Hitable **list, int n, float time0, float time1);
  
  virtual bool hit(const Ray& r, float tmin, float tmax, hit_record& rec) const;
  virtual bool bounding_box(float t0, float t1, aabb &box) const;
  
  Hitable *left;  // Left child
  Hitable *right; // Right child
  
  aabb box;
  
};


#endif /* BVH_node_HH_INCLUDE */
