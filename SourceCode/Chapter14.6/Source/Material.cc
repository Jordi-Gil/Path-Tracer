#include "Material.hh"

float schlick(float cosine, float ref_idx){
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0) * pow((1-cosine), 5);
}

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

Vector3 reflect(const Vector3& v, const Vector3& n){
    return v - 2.0*dot(v,n)*n;
}

Vector3 random_in_unit_sphere(){
  Vector3 p;
  do{
    p = 2.0*Vector3((rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0))) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return p;
}

Material::Material(int t, const Vector3 &a, float f, float ri) {
  type = t;
  albedo = a;
  fuzz = f;
  ref_idx = ri;
}

bool Material::scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray &scattered) {
  
  if(type == LAMBERTIAN) return Lambertian(rec, attenuation, scattered);
  else if(type == METAL) return Metal(r_in, rec, attenuation, scattered);
  else if(type == DIELECTRIC) return Dielectric(r_in, rec, attenuation, scattered);
  else if(type == DIFFUSE_LIGHT) return false;
  else return false;
  
}

Vector3 Material::emitted() {
  if(type == DIFFUSE_LIGHT) return albedo;
  else return Vector3::Zero();
}

bool Material::Lambertian(const hit_record &rec, Vector3 &attenuation, Ray& scattered) {
    
    Vector3 target = rec.point + rec.normal + random_in_unit_sphere();
    
    scattered = Ray(rec.point, target-rec.point);
    attenuation = albedo;
    
    return true;
}

bool Material::Metal(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered) {
    
    Vector3 reflected = reflect( unit_vector( r_in.direction()), rec.normal);
    
    scattered = Ray(rec.point, reflected + fuzz*random_in_unit_sphere());
    attenuation = albedo;
    
    return (dot(scattered.direction(), rec.normal) > 0);
    
}

bool Material::Dielectric(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered) {
    
    Vector3 outward_normal;
    Vector3 reflected = reflect(r_in.direction(), rec.normal);
    
    float ni_over_nt;
    attenuation = albedo;
    Vector3 refracted;
    float reflect_prob;
    float cosine;
    if(dot(r_in.direction(), rec.normal) > 0){
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0 / ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    
    if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
        reflect_prob = schlick(cosine, ref_idx);
    }
    else{
        scattered = Ray(rec.point, reflected);
        reflect_prob = 1.0;
    }
    
    if((rand()/(RAND_MAX + 1.0)) < reflect_prob){
        scattered = Ray(rec.point, reflected);
    }
    else{
        scattered = Ray(rec.point, refracted);
    }
    return true;
    
}

const char *Material::getName() {
  if(type == LAMBERTIAN) return "Lambertian";
  else if(type == METAL) return "Metal";
  else if(type == DIELECTRIC) return "Dielectric";
  else return "none";
}

Vector3 Material::getAlbedo() {
  return albedo;
}
