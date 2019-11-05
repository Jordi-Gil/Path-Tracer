#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.cuh"

#include <curand.h>
#include <curand_kernel.h>

class Camera {

public:
    
    __host__ __device__ Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float time0, float time1);
    __device__ Ray get_ray(float s, float t, curandState *random);
    
    Vector3 origin;
    Vector3 lower_left_corner;
    Vector3 horizontal;
    Vector3 vertical;
    Vector3 w,u,v;
    float time0, time1;
    float lens_radius;
};

#endif /* CAMERA_HH_INCLUDE */
