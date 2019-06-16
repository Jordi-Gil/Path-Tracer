#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.cuh"

class Camera {

public:
    
    __host__ __device__ Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist);
    __device__ Ray get_ray(float s, float t);
    
    Vector3 origin;
    Vector3 lower_left_corner;
    Vector3 horizontal;
    Vector3 vertical;
    Vector3 w,u,v;
    float lens_radius;
};

#endif /* CAMERA_HH_INCLUDE */
