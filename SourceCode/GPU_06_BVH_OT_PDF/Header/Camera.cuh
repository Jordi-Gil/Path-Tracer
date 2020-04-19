#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.cuh"

#include <curand.h>
#include <curand_kernel.h>

class Camera {

public:
    
  __host__ __device__ Camera() {}
  __host__ __device__ Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float time0, float time1);
  __device__ Ray get_ray(float s, float t, curandState *random);

  __host__ __device__ Vector3 getLookfrom(){ return lookfrom; }
  __host__ __device__ Vector3 getLookat() {return lookat;}
  __host__ __device__ Vector3 getVUP() {return vup;}

  __host__ __device__ float getFOV(){return vfov;}
  __host__ __device__ float getAspect() {return aspect;}
  __host__ __device__ float getAperture(){return aperture;}
  __host__ __device__ float getFocus(){return focus_dist;}

  Vector3 origin;
  Vector3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 w,u,v;
  float time0, time1;
  float lens_radius;

private:

  Vector3 lookfrom, lookat, vup;
  float vfov, aspect, aperture, focus_dist;

};

#endif /* CAMERA_HH_INCLUDE */
