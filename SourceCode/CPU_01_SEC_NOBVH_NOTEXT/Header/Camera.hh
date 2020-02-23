#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.hh"

class Camera {

public:
    
  Camera() {}
  Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1);
  Ray get_ray(float s, float t);
  
  
  Vector3 eyePos;
  Vector3 lookAt;
  
  Vector3 origin;
  Vector3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 w,u,v;
  float time0, time1;
  float lens_radius;
};

#endif /* CAMERA_HH_INCLUDE */
