#include "Camera.hh"

Vector3 random_in_unit_disk(){
    Vector3 p;
    do{
        p = 2.0*Vector3((rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0)), 0) - Vector3(1,1,0);
    }while(dot(p,p) >= 1.0);
    return p;
}

Camera::Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {
  
  time0 = t0;
  time1 = t1;
  eyePos = lookfrom;
  
  lens_radius = aperture / 2;
  float theta = vfov * M_PI/180.0;
  
  float half_width = 2 * tan(theta/2);
  float half_height = aspect * half_width;
  
  w = unit_vector(lookfrom - lookat);
  u = cross(vup, w);
  v = cross(w, u);
  
  u = unit_vector(u);
  v = unit_vector(v);
  lower_left_corner = eyePos - half_width/2. * focus_dist * u - half_height/2. * focus_dist * v - focus_dist * w;
  
  horizontal = half_width * focus_dist * u;
  vertical = half_height * focus_dist * v;
  
  origin = lower_left_corner - eyePos;
}

Ray Camera::get_ray(float s, float t) {
  
  float time = time0 + (rand()/(RAND_MAX + 1.0)) * (time1-time0);
  
  if(lens_radius == 0.) return Ray(eyePos, origin + s * horizontal + t * vertical, time);
  
  Vector3 rd = lens_radius * random_in_unit_disk();
  Vector3 offset = rd.x() * u + rd.y() * v;
  
  return Ray(eyePos + offset, 
              origin + s * horizontal + t * vertical - offset, time);
}
