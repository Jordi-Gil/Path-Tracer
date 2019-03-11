#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.hh"

class Camera {

public:
    
    Camera();
    Ray get_ray(float u, float v);
    
    Vector3 origin;
    Vector3 lower_left_corner;
    Vector3 horizontal;
    Vector3 vertical;

};

#endif /* CAMERA_HH_INCLUDE */
