#include "Material.cuh"

__host__ __device__ float schlick(float cosine, float ref_idx){
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0) * pow((1-cosine), 5);
}

__host__ __device__ bool refract(const Vector3 &v, const Vector3 &n, float ni_over_nt, Vector3 &refracted) {
  
  Vector3 uv = unit_vector(v);
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
    p = 2.0*Vector3(curand_uniform(random), 
                    curand_uniform(random),
                    curand_uniform(random)
                   ) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return p;
}

__device__ Vector3 random_on_unit_sphere(curandState *random){
  Vector3 p;
  do{
    p = 2.0*Vector3(curand_uniform(random), 
                    curand_uniform(random),
                    curand_uniform(random)
                   ) - Vector3::One();
  }
  while(p.squared_length() >= 1.0);
  return unit_vector(p);
}

__device__ inline Vector3 random_cosine_direction(curandState *random){
  float r1 = curand_uniform(random);
  float r2 = curand_uniform(random);
  float z = sqrt(1-r2);
  float phi = 2*M_PI*r1;
  float x = cos(phi)*sqrt(r2);
  float y = sin(phi)*sqrt(r2);
  
  return Vector3(x,y,z);
}

__host__ __device__ Material::Material(int t, Texture a, float f, float ri, const Vector3 &_attenuation) {
  
  type = t;
  albedo = a;
  fuzz = f;
  ref_idx = ri;
  
}

__device__ bool Material::scatter(const Ray& r_in, const hit_record &rec, scatter_record &srec, curandState *random, bool oneTex, unsigned char **d_textures) {
    
  if(type == LAMBERTIAN) return Lambertian(r_in, rec, srec, random, oneTex, d_textures);
  else if (type == METAL) return Metal(r_in, rec, srec, random,oneTex, d_textures);
  else if (type == DIELECTRIC) return Dielectric(r_in, rec, srec, random,oneTex, d_textures);
  else if(type == DIFFUSE_LIGHT) return false;
  else return false;
  
}

__device__ float Material::scatter_pdf(const Ray& r_in, const hit_record &rec, Ray& scattered) {
    
  if(type == LAMBERTIAN) return Lambertian_pdf(r_in, rec, scattered);/*
  else if (type == METAL) return Metal(r_in, rec, attenuation, scattered, random,oneTex, d_textures);
  else if (type == DIELECTRIC) return Dielectric(r_in, rec, attenuation, scattered, random,oneTex, d_textures);
  else if(type == DIFFUSE_LIGHT) return false;*/
  else return 0;
  
}

__device__ Vector3 Material::emitted(float u, float v, bool oneTex, unsigned char **d_textures) {
  if(type == DIFFUSE_LIGHT || type == SKYBOX) return albedo.value(u,v,oneTex,d_textures);
  else return Vector3::Zero();
}

__device__ bool Material::Lambertian(const Ray& r_in, const hit_record &hrec, scatter_record &srec, curandState *random, bool oneTex, unsigned char **d_textures) {
  
  srec.is_specular = false;
  srec.attenuation = albedo.value(hrec.u, hrec.v,oneTex,d_textures);
  srec.pdf_ptr = new pdf(COSINE,hrec.normal);
  
  return true;
}

__device__ float Material::Lambertian_pdf(const Ray& r_in, const hit_record &rec, Ray &scattered){
  
  float cos = dot(rec.normal, unit_vector(scattered.direction()));
  
  if(cos < 0) return 0;
  return cos / M_PI;
  
}

__device__  bool Material::Metal(const Ray& r_in, const hit_record& hrec, scatter_record &srec, curandState *random, bool oneTex, unsigned char **d_textures) {

  
  Vector3 reflected = reflect( unit_vector( r_in.direction()), hrec.normal);

  srec.specular_ray = Ray(hrec.point, reflected + fuzz*random_in_unit_sphere(random));
  srec.attenuation = albedo.value(hrec.u, hrec.v, oneTex, d_textures);
  srec.pdf_ptr = 0;
  srec.is_specular = true;

  return true;
    
}

__device__ bool Material::Dielectric(const Ray& r_in, const hit_record& hrec, scatter_record &srec, curandState *random, bool oneTex, unsigned char **d_textures) {

  srec.is_specular = true;
  
  Vector3 outward_normal;
  Vector3 reflected = reflect(r_in.direction(), hrec.normal);
  
  float ni_over_nt;
  
  srec.attenuation = albedo.value(hrec.u, hrec.v, oneTex, d_textures);
  
  Vector3 refracted;
  float reflect_prob;
  float cosine;
  
//   bool inside = true;
  
  if(dot(r_in.direction(), hrec.normal) > 0){
    outward_normal = -hrec.normal;
    ni_over_nt = ref_idx;
    cosine = ref_idx * dot(r_in.direction(), hrec.normal) / r_in.direction().length();
//     inside = true;
  }
  else {
    outward_normal = hrec.normal;
    ni_over_nt = 1.0 / ref_idx;
    cosine = -dot(r_in.direction(), hrec.normal) / r_in.direction().length();
  }
    
  if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
    reflect_prob = schlick(cosine, ref_idx);
  }
  else{
    reflect_prob = 1.0;
  }
  
//   if(inside) {
//     float distance = (hrec.point - r_in.point_at_parameter(0)).length();
//     srec.attenuation = albedo.value(hrec.u, hrec.v, oneTex, d_textures) * attenuation;
//   }
  
  if(curand_uniform(random) < reflect_prob){
    srec.specular_ray = Ray(hrec.point, reflected);
  }
  else{
    srec.specular_ray = Ray(hrec.point, refracted);
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
