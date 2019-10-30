#ifndef _MATERIAL_HH_INCLUDE
#define _MATERIAL_HH_INCLUDE

#include "Ray.hh"

struct hit_record;

enum type {
  LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT
};

class Material {

public:
    
  Material() {}
  Material(int t, const Vector3 &a = Vector3::One(), float f = -1.0, float ri = -1.0);
  
  bool scatter(const Ray& r_in, const hit_record &rec, Vector3& attenuation, Ray& scattered);
  Vector3 emitted();
  
  bool Lambertian(const hit_record &rec, Vector3 &attenuation, Ray &scattered);
  bool Metal(const Ray &r_in, const hit_record &rec, Vector3 &attenuation, Ray &scattered);
  bool Dielectric(const Ray &r_in, const hit_record &rec, Vector3 &attenuation, Ray &scattered);
  
  Vector3 getAlbedo();
  const char *getName();
  
  Vector3 albedo = Vector3::One();
  float fuzz;
  float ref_idx;
  int type;
    
};

struct hit_record {
  float t;
  Vector3 point;
  Vector3 normal;
  Material mat_ptr;
};

#endif /* _MATERIAL_HH_INCLUDE */
