#include "Material.hh"

bool refract(const Vector3 &v, const Vector3 &n, float ni_over_nt, Vector3 &refracted){
    Vector3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    
    if(discriminant > 0){
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        
        return true;
    }
    else return false;
}

Vector3 reflect(const Vector3 &v, const Vector3& n){
    return v - 2*dot(v,n)*n;
}

Vector3 random_in_unit_sphere(){
  Vector3 p;
  do{
    p = 2.0*Vector3((rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0))) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return p;
}

bool Lambertian::scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered) const {
    
    Vector3 target = rec.point + rec.normal + random_in_unit_sphere();
    
    scattered = Ray(rec.point, target-rec.point);
    attenuation = albedo;
    
    return true;
}


bool Metal::scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered) const {
    
    Vector3 reflected = reflect( unit_vector( r_in.direction()), rec.normal);
    
    scattered = Ray(rec.point, reflected + fuzz*random_in_unit_sphere());
    attenuation = albedo;
    
    return (dot(scattered.direction(), rec.normal) > 0);
    
}

bool Dielectric::scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered) const{
    
    Vector3 outward_normal;
    Vector3 reflected = reflect(r_in.direction(), rec.normal);
    
    float ni_over_nt;
    attenuation = Vector3(1.0,1.0,0.0);
    Vector3 refracted;
    
    if(dot(r_in.direction(), rec.normal) > 0){
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0 / ref_idx;
    }
    
    if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
        scattered = Ray(rec.point, refracted);
    }
    else{
        scattered = Ray(rec.point, reflected);
        return false;
    }
    return true;
    
}
