#ifndef _MATERIAL_HH_INCLUDE
#define _MATERIAL_HH_INCLUDE

#include <curand.h>
#include <curand_kernel.h>

#include "Ray.cuh"
#include "Texture.cuh"
#include "ONB.cuh"
#include "pdf.cuh"

struct hit_record;
struct scatter_record;

enum type {
  LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT, SKYBOX
};

class Material {

public:
  
  __host__ __device__ Material() {}
  __host__ __device__ Material(int t, Texture a, float f = -1.0, float ri = -1.0, const Vector3 &_attenuation = Vector3::Zero());  
  
  __device__ bool scatter(const Ray& r_in, const hit_record& rec, scatter_record &srec, curandState *random, bool oneTex = false, unsigned char **d_textures = 0);
  __device__ float scatter_pdf(const Ray& r_in, const hit_record &rec, Ray& scattered);
  
  __device__ Vector3 emitted(float u, float v, bool oneTex = false, unsigned char **d_textures = 0);
  
  __device__ bool Lambertian(const Ray& r_in, const hit_record &rec, scatter_record &srec, curandState *random, bool oneTex = false, unsigned char **d_textures = 0);
  __device__ float Lambertian_pdf(const Ray& r_in, const hit_record &rec, Ray &scattered);
  __device__ bool Metal(const Ray& r_in, const hit_record& rec, scatter_record &srec, curandState *random, bool oneTex = false, unsigned char **d_textures = 0);
  __device__ bool Dielectric(const Ray& r_in, const hit_record& rec, scatter_record &srec, curandState *random, bool oneTex = false, unsigned char **d_textures = 0);
  
  __host__ __device__ const char *getName();
  
  __host__ void hostToDevice(int numGPUs){ albedo.hostToDevice(numGPUs); }

  Texture albedo;
  float fuzz;
  float ref_idx;
  int type;
  Vector3 attenuation;

};

struct hit_record {
  float t;
  float u;
  float v;
  Vector3 point;
  Vector3 normal;
  Material mat_ptr;
};

struct scatter_record {
  Ray specular_ray;
  bool is_specular;
  Vector3 attenuation;
  pdf *pdf_ptr;
};


#endif /* _MATERIAL_HH_INCLUDE */
