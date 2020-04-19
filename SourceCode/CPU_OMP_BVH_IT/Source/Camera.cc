#include "Camera.hh"

Vector3 random_in_unit_disk(){
    Vector3 p;
    do{
        p = 2.0*Vector3((rand()/(RAND_MAX + 1.0)), (rand()/(RAND_MAX + 1.0)), 0) - Vector3(1,1,0);
    }while(dot(p,p) >= 1.0);
    return p;
}

Camera::Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float time0, float time1){
  
  this->lookat = lookat;
  this->lookfrom = lookfrom;
  this->vup = vup;
  this->aspect = aspect;
  this->aperture = aperture;
  this->focus_dist = focus_dist;
  this->time0 = time0;
  this->time1 = time1;
    
  lens_radius = aperture / 2;
  float theta = vfov*M_PI/180.0;
  float half_height = tan(theta/2);
  float half_width = aspect * half_height;
  origin = lookfrom;
  w = normalize(lookfrom - lookat);
  u = normalize(cross(vup, w));
  v = cross(w, u);
    
  lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
  horizontal = 2*half_width*focus_dist*u;
  vertical = 2*half_height*focus_dist*v;
}

Ray Camera::get_ray(float s, float t){
    Vector3 rd = lens_radius*random_in_unit_disk();
    Vector3 offset = u*rd.x() + v*rd.y();
    float time = time0 + (rand()/(RAND_MAX + 1.0)) * (time1-time0);
    return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset, time);
}
