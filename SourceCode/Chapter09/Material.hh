#ifndef _MATERIAL_HH_INCLUDE
#define _MATERIAL_HH_INCLUDE

struct hit_record;

#include "Ray.hh"
#include "Hitable.hh"


class Material {

public:
    
    virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered) const = 0;

};

class Lambertian: public Material {
    
public:
    Lambertian(const Vector3& a) : albedo(a) {}
    virtual bool scatter(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered) const;
    
    Vector3 albedo;
};


class Metal: public Material{
    
public:
    Metal(const Vector3& a, float f) : albedo(a) { if(f < 1) fuzz = f; else fuzz = 1; }
    virtual bool scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered) const;
    
    Vector3 albedo;
    float fuzz;
};


class Dielectric: public Material{
  
public:
    Dielectric(float ri) : ref_idx(ri) {}
    virtual bool scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered) const;
    float ref_idx;
};
#endif /* _MATERIAL_HH_INCLUDE */
