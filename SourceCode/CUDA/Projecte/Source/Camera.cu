#include "Camera.cuh"

__host__ __device__ Camera::Camera(Vector3 lookfrom, Vector3 lookat, Vector3 vup, float vfov, float aspect, float aperture, float focus_dist){
  
    Vector3 w,u,v;
    float theta = vfov*M_PI/180.0;
    float half_height = tan(theta/2);
    float half_width = aspect * half_height;
    origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);
    
    lower_left_corner = Vector3(-half_width, -half_height, -1.0);
    lower_left_corner = origin - half_width*u - half_height*v - w;
    horizontal = 2*half_width*u;
    vertical = 2*half_height*v;
}


__device__ Ray Camera::get_ray(float u, float v){
    return Ray(origin, 
               lower_left_corner + u*horizontal + v*vertical - origin);
}


