#ifndef _MATERIAL_HH_INCLUDE
#define _MATERIAL_HH_INCLUDE

#include <curand.h>
#include <curand_kernel.h>

#include "Ray.cuh"

struct hit_record;

enum type {
  LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT
};

class Material {

public:
  
  __host__ __device__ Material() {}
  __host__ __device__ Material(int t, const Vector3 &a = Vector3::One(), float f = -1.0, float ri = -1.0);  
  
  __device__ bool scatter(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random);
  __device__ Vector3 emitted();
  
  __device__ bool Lambertian(const hit_record &rec, Vector3 &attenuation, Ray &scattered, curandState *random);
  __device__ bool Metal(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random);
  __device__ bool Dielectric(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random);
  
  __host__ __device__ Vector3 getAlbedo();
  __host__ __device__ const char *getName();

  Vector3 albedo = Vector3::One();
  float fuzz;
  float ref_idx;
  int type;

};

struct hit_record{

  float t;
  Vector3 point;
  Vector3 normal;
  Material mat_ptr;
};

#endif /* _MATERIAL_HH_INCLUDE */
