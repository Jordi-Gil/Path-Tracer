#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.hh"

class Camera {

public:
    
    Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect);
    Ray get_ray(float s, float t);
    
    Vector3 origin;
    Vector3 lower_left_corner;
    Vector3 horizontal;
    Vector3 vertical;

};

#endif /* CAMERA_HH_INCLUDE */
