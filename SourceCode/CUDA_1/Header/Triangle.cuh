#ifndef _TRIANGLE_HH_INCLUDE
#define _TRIANGLE_HH_INCLUDE

#include <stdexcept>
#include <assert.h>

#include "Material.cuh"
#include "Math.cuh"

class Triangle {
  
public:
    
  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(Vector3 v1, Vector3 v2, Vector3 v3, Material mat, Vector3 t = Vector3::One());
  __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, hit_record& rec);
  
  __host__ __device__ Vector3 operator[](int i) const;
  __host__ __device__ Vector3& operator[](int i);
  __host__ __device__ Vector3 getCentroid();
  __host__ __device__ Material getMaterial();
  
private:
  
  Vector3 vertex[3];
  Vector3 centroid;
  Material mat_ptr;
  Vector3 uv;
  
};

#endif /* _TRIANGLE_HH_INCLUDE */
