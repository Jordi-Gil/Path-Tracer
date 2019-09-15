#ifndef _MATERIAL_HH_INCLUDE
#define _MATERIAL_HH_INCLUDE

struct hit_record;

#include <curand.h>
#include <curand_kernel.h>

#include "Ray.cuh"
#include "Hitable.cuh"

class Material {

public:
    
    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random) const = 0;

	Vector3 albedo = Vector3::One();

};

class Lambertian: public Material {
    
public:
    __device__ Lambertian(const Vector3& a) : albedo(a) {}
    __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random) const;
    
    Vector3 albedo;
};


class Metal: public Material{
    
public:
    __device__ Metal(const Vector3& a, float f) : albedo(a) { if(f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered, curandState *random) const;
    
    Vector3 albedo;
    float fuzz;
};


class Dielectric: public Material{
  
public:
    __device__ Dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered, curandState *random) const;
    
    float ref_idx;
};
#endif /* _MATERIAL_HH_INCLUDE */
