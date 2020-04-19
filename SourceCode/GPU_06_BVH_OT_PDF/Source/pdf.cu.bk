#include "pdf.cuh"

__device__ inline Vector3 random_cosine_direction(curandState *_random){
  float r1 = curand_uniform(_random);
  float r2 = curand_uniform(_random);
  float z = sqrt(1-r2);
  float phi = 2*M_PI*r1;
  float x = cos(phi)*sqrt(r2);
  float y = sin(phi)*sqrt(r2);
  
  return Vector3(x,y,z);
}

__host__ __device__ pdf::pdf(int _type, const Vector3 &w,/* ShapeList* p,*/ const Vector3 &o) {
  
  type = _type;
  
  if(type == COSINE){
    
    uvw.build_from_w(w);
    
  }
//   else if (type == SHAPE){
//     ptr = p;
//     origin = o;
//   }
  
}
  
__device__ Vector3 pdf::generate(curandState *nrandom){
  
  if(type == COSINE) return generate_cosine(nrandom);
//   else if (type == SHAPE) return generate_shape(_random);
  else return Vector3::Zero();
  
}

__device__ Vector3 pdf::generate_cosine(curandState *_random) {
  return uvw.local(random_cosine_direction(_random));
}

// __device__ Vector3 pdf::generate_shape(curandState *_random) {
//   return ptr->random(origin, _random);
// }

__device__ float pdf::value(const Vector3 &direction){
  
  if(type == COSINE) return cosine_value(direction);
//   else if(type == SHAPE) return shape_value(direction);
  else return 0;
  
}
  
__device__ float pdf::cosine_value(const Vector3 &direction) {
  
  float cosine = dot(unit_vector(direction), uvw.w());
  
  if(cosine > 0) return cosine/M_PI;
  else return 0;
  
}

// __device__ float pdf::shape_value(const Vector3 &direction){
//   return ptr->pdf_value(origin, direction);
// }
