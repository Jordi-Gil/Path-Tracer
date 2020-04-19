#ifndef CAMERA_HH_INCLUDE
#define CAMERA_HH_INCLUDE

#include "Ray.hh"

class Camera {

public:

  Camera() {}
  Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float time0, float time1);
  Ray get_ray(float s, float t);
    
  Vector3 getLookfrom(){ return lookfrom; }
  Vector3 getLookat() {return lookat;}
  Vector3 getVUP() {return vup;}

  float getFOV(){return vfov;}
  float getAspect() {return aspect;}
  float getAperture(){return aperture;}
  float getFocus(){return focus_dist;}
  
private:
    
  Vector3 origin;
  Vector3 lower_left_corner;
  Vector3 horizontal;
  Vector3 vertical;
  Vector3 w,u,v;
  float time0, time1;
  float lens_radius;

  Vector3 lookfrom, lookat, vup;
  float vfov, aspect, aperture, focus_dist;
};

#endif /* CAMERA_HH_INCLUDE */
