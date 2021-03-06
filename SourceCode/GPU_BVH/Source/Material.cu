#include "Material.cuh"

__host__ __device__ float schlick(float cosine, float ref_idx){
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0) * pow((1-cosine), 5);
}

__host__ __device__ bool refract(const Vector3 &v, const Vector3 &n, float ni_over_nt, Vector3 &refracted) {
  
  Vector3 uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);

  if(discriminant > 0){
    refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
    return true;
  }
  else return false;
}

__host__ __device__ Vector3 reflect(const Vector3& v, const Vector3& n){
  return v - 2.0*dot(v,n)*n;
}

__device__ Vector3 random_in_unit_sphere(curandState *random){
  Vector3 p;
  do{
    p = 2.0*Vector3(curand_uniform(random), curand_uniform(random), curand_uniform(random)) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return p;
}

__device__ Vector3 random_on_unit_sphere(curandState *random){
  Vector3 p;
  do{
    p = 2.0*Vector3(curand_uniform(random), curand_uniform(random), curand_uniform(random)) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return normalize(p);
}

__host__ __device__ Material::Material(int t, Texture a, float f, float ri) {
  
  type = t;
  albedo = a;
  fuzz = f;
  ref_idx = ri;
  
}

__device__ bool Material::scatter(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray& scattered, curandState *random, bool oneTex, unsigned char **d_textures) {
    
  if(type == LAMBERTIAN) return Lambertian(r_in, rec, attenuation, scattered, random, oneTex, d_textures);
  else if (type == METAL) return Metal(r_in, rec, attenuation, scattered, random,oneTex, d_textures);
  else if (type == DIELECTRIC) return Dielectric(r_in, rec, attenuation, scattered, random,oneTex, d_textures);
  else if(type == DIFFUSE_LIGHT) return false;
  else return false;
  
}

__device__ Vector3 Material::emitted(float u, float v, bool oneTex, unsigned char **d_textures) {
  if(type == DIFFUSE_LIGHT || type == SKYBOX) return albedo.value(u, v, oneTex, d_textures);
  else return Vector3::Zero();
}

__device__ bool Material::Lambertian(const Ray& r_in, const hit_record &rec, Vector3 &attenuation, Ray &scattered, curandState *random, bool oneTex, unsigned char **d_textures) {

  Vector3 target = rec.point + rec.normal + random_in_unit_sphere(random);
  
  scattered = Ray(rec.point, target-rec.point, r_in.time());
  attenuation = albedo.value(rec.u, rec.v, oneTex, d_textures);
  
  return true;
}

__device__  bool Material::Metal(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random, bool oneTex, unsigned char **d_textures) {

  Vector3 reflected = reflect( normalize( r_in.direction()), rec.normal);

  scattered = Ray(rec.point, reflected + fuzz*random_in_unit_sphere(random), r_in.time());
  attenuation = albedo.value(rec.u, rec.v, oneTex, d_textures);

  return (dot(scattered.direction(), rec.normal) > 0);
    
}

__device__ bool Material::Dielectric(const Ray& r_in, const hit_record& rec, Vector3& attenuation, Ray& scattered, curandState *random, bool oneTex, unsigned char **d_textures) {

  Vector3 outward_normal;
  Vector3 reflected = reflect(r_in.direction(), rec.normal);
  
  float ni_over_nt;
  attenuation = albedo.value(rec.u, rec.v, oneTex, d_textures);
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
    
  if(curand_uniform(random) < reflect_prob){
    scattered = Ray(rec.point, reflected);
  }
  else{
    scattered = Ray(rec.point, refracted);
  }
  return true;
}

__host__ __device__ const char *Material::getName(){
  
  if(type == LAMBERTIAN) return "Lambertian";
  else if (type == METAL) return "Metal";
  else if (type == DIELECTRIC) return "Dielectric";
  else if (type == DIFFUSE_LIGHT) return "Diffuse Light";
  else return "none";
  
}
